"""
Regression tests for the reduction pipeline.

These tests re-run the reduction from raw FITS files and compare the output
with the existing reduced NPZ (which is the golden reference — no separate
golden generation step is needed, the NPZ files already on Narval ARE the
reference).

Requirements:
    - Raw FITS files accessible at obs_dir (Narval or local)
    - Pipeline config YAML for each dataset
    - ~/.starships/regression_config.yaml with a 'reduction_datasets' section

These tests are SLOW (5-15 min per night) and require raw data, so they are
opt-in only. Run with:

    pytest tests/regression/ -v -m reduction

They are automatically skipped in all other contexts.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Config and markers
# ---------------------------------------------------------------------------

_REGRESSION_CONFIG = Path.home() / '.starships' / 'regression_config.yaml'
_config_available  = _REGRESSION_CONFIG.exists()


def _reduction_dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('reduction_datasets', {}).keys())


pytestmark = pytest.mark.reduction  # entire module requires -m reduction


requires_reduction_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "Reduction config not found. "
        "On Narval: add 'reduction_datasets' to ~/.starships/regression_config.yaml"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def reduction_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReductionPipeline:
    """
    Re-run the reduction pipeline from raw FITS files and verify the output
    matches the existing reduced NPZ files.

    The comparison covers:
      - flux        (PCA-subtracted, normalized spectra)
      - wave        (wavelength solution)
      - noise       (per-pixel noise estimate)
      - PCA components (ensure the same systematics are removed)

    Tolerance: rtol=1e-8 (slightly looser than logL tests because the PCA
    decomposition can have minor floating-point ordering differences, though
    in practice it should be bit-identical on the same machine).
    """

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_reduced_flux_unchanged(self, reduction_config, ds_name):
        """Re-reduced flux must match the golden NPZ."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        np.testing.assert_allclose(
            transit.flux.data, golden['flux'],
            rtol=1e-8,
            err_msg=f"[{ds_name}] Reduced flux differs from golden NPZ",
        )

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_reduced_wave_unchanged(self, reduction_config, ds_name):
        """Re-reduced wavelength solution must match the golden NPZ."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        np.testing.assert_allclose(
            transit.wave, golden['wave'],
            rtol=1e-10,
            err_msg=f"[{ds_name}] Wavelength solution differs from golden NPZ",
        )

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_reduced_noise_unchanged(self, reduction_config, ds_name):
        """Re-reduced noise array must match the golden NPZ."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        np.testing.assert_allclose(
            transit.noise.data, golden['noise'],
            rtol=1e-8,
            err_msg=f"[{ds_name}] Noise array differs from golden NPZ",
        )

    @requires_reduction_data
    @pytest.mark.parametrize("ds_name", _reduction_dataset_names())
    def test_pca_components_unchanged(self, reduction_config, ds_name):
        """PCA components must match — same systematics are removed."""
        transit, golden = self._reduce_and_load_golden(reduction_config, ds_name)
        np.testing.assert_allclose(
            transit.pca.components_, golden['components_'],
            rtol=1e-8,
            err_msg=f"[{ds_name}] PCA components differ from golden NPZ",
        )

    # ------------------------------------------------------------------
    # Shared reduction helper (cached per test session via lru_cache-like
    # pattern — reduction is expensive, run once per dataset name)
    # ------------------------------------------------------------------

    _cache: dict = {}

    def _reduce_and_load_golden(self, reduction_config, ds_name):
        """Run reduction once per dataset and cache the result."""
        if ds_name not in self._cache:
            self._cache[ds_name] = _run_reduction(reduction_config, ds_name)
        return self._cache[ds_name]


# ---------------------------------------------------------------------------
# Reduction runner (module-level so it can be used standalone)
# ---------------------------------------------------------------------------

def _run_reduction(reduction_config, ds_name):
    """Re-run the reduction following the notebook code path and return (transit, golden_data)."""
    import yaml as _yaml
    import astropy.units as u
    import astropy.constants as const
    import starships.planet_obs as pl_obs
    from starships.planet_obs import Observations
    from pipeline.reduction import pl_param_units

    ds_cfg = reduction_config['reduction_datasets'][ds_name]

    # Load the pipeline config YAML for this dataset
    pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
    if not pipeline_cfg_path.exists():
        pytest.skip(f"Pipeline config not found: {pipeline_cfg_path}")
    with open(pipeline_cfg_path) as f:
        config_dict = _yaml.safe_load(f)

    # Check raw FITS are accessible
    obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
    if not obs_dir.exists():
        pytest.skip(f"Raw data directory not found: {obs_dir}")

    # Check golden NPZ exists
    golden_path = Path(ds_cfg['golden_npz']).expanduser()
    if not golden_path.exists():
        pytest.skip(f"Golden NPZ not found: {golden_path}")

    # Build planet kwargs from config (skip null values, same as retrieval.py)
    pl_kwargs = pl_param_units(config_dict) if config_dict.get('pl_params') else {}

    # Create Observations and load raw data (notebook code path: CADC=False by default)
    visit_name = ds_cfg['visit_name']
    list_filenames = {
        'list_e2ds':  f'list_e2ds_{visit_name}',
        'list_tcorr': f'list_tcorr_{visit_name}',
        'list_recon': f'list_recon_{visit_name}',
    }

    print(f"\n  [{ds_name}] Loading raw data from {obs_dir} (visit: {visit_name}) ...")
    obs = Observations(name=config_dict['pl_name'], pl_kwargs=pl_kwargs)
    obs.fetch_data(obs_dir, **list_filenames)
    obs.n_spec = len(obs.filenames)  # not set automatically by fetch_data

    # All exposures; remove bad ones if specified in config
    all_exp = np.arange(obs.n_spec)
    bad = config_dict.get('bad_indexs', {}).get(visit_name, [])
    transit_tags = [np.delete(all_exp, bad) if bad else all_exp]

    # Reduction parameters (notebook format)
    n_pc       = ds_cfg['n_pc']
    mask_tellu = ds_cfg['mask_tellu']
    mask_wings = ds_cfg['mask_wings']
    params_all = [[mask_tellu, mask_wings, 51, 41, 5, n_pc, 5.0, 5.0, 5.0, 5.0]]

    kwargs_gen_tr = {
        'coeffs':     config_dict['coeffs'],
        'ld_model':   config_dict['ld_model'],
        'do_tr':      [1],
        'kind_trans': config_dict['kind_trans'],
        'polynome':   [False],
        'cbp':        True,
    }
    kwargs_build_ts = {
        'clip_ratio':  config_dict['clip_ratio'],
        'clip_ts':     config_dict['clip_ts'],
        'unberv_it':   config_dict['unberv_it'],
    }

    print(f"  [{ds_name}] Running reduction "
          f"(n_pc={n_pc}, mask_tellu={mask_tellu}, mask_wings={mask_wings}) ...")
    list_tr = pl_obs.generate_all_transits(
        obs, transit_tags, [0.0], params_all, config_dict['iout_all'],
        counting=False, **kwargs_gen_tr, **kwargs_build_ts,
    )
    transit = list_tr['1']

    golden = np.load(golden_path, allow_pickle=True)
    return transit, golden
