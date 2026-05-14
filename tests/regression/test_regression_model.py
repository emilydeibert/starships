"""
Regression tests for petitRADTRANS model generation.

These tests re-run prepare_model_multi_reg with a fixed set of parameters and
compare the output (wavelength grid + model spectrum) against a golden NPZ.

Requirements:
    - petitRADTRANS installed and opacity tables accessible
    - Pipeline config YAML for each dataset (retrieval input file)
    - ~/.starships/regression_config.yaml with a 'model_datasets' section

These tests require petitRADTRANS on Narval and are opt-in only. Run with:

    pytest tests/regression/ -v -m model

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


def _model_dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('model_datasets', {}).keys())


pytestmark = pytest.mark.model  # entire module requires -m model


requires_model_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "Model config not found. "
        "On Narval: add 'model_datasets' to ~/.starships/regression_config.yaml"
    ),
)

_prt_available = False
try:
    import petitRADTRANS  # noqa: F401
    _prt_available = True
except ImportError:
    pass

requires_prt = pytest.mark.skipif(
    not _prt_available,
    reason="petitRADTRANS is not installed on this system",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def model_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelGeneration:
    """
    Re-run prepare_model_multi_reg with fixed parameters and verify the output
    matches the golden NPZ generated on Narval.

    The comparison covers:
      - wv   (wavelength grid in µm)
      - model (model spectrum flux ratio or emission)

    Tolerance: rtol=1e-5 (model generation involves petitRADTRANS radiative
    transfer which may show small numerical differences across platforms, but
    should be stable on the same machine with the same opacity tables).
    """

    @requires_model_data
    @requires_prt
    @pytest.mark.parametrize("ds_name", _model_dataset_names())
    def test_model_wavelength_grid_unchanged(self, model_config, ds_name):
        """Re-generated wavelength grid must match the golden NPZ."""
        wv_new, _ = self._generate_and_load_golden(model_config, ds_name)
        golden_path = _get_golden_path(model_config, ds_name)
        golden = np.load(golden_path, allow_pickle=True)
        np.testing.assert_allclose(
            wv_new, golden['wv'],
            rtol=1e-10,
            err_msg=f"[{ds_name}] Wavelength grid differs from golden NPZ",
        )

    @requires_model_data
    @requires_prt
    @pytest.mark.parametrize("ds_name", _model_dataset_names())
    def test_model_spectrum_unchanged(self, model_config, ds_name):
        """Re-generated model spectrum must match the golden NPZ."""
        wv_new, model_new = self._generate_and_load_golden(model_config, ds_name)
        golden_path = _get_golden_path(model_config, ds_name)
        golden = np.load(golden_path, allow_pickle=True)
        np.testing.assert_allclose(
            model_new, golden['model'],
            rtol=1e-5,
            err_msg=f"[{ds_name}] Model spectrum differs from golden NPZ",
        )

    # ------------------------------------------------------------------
    # Shared generation helper (cached per test session)
    # ------------------------------------------------------------------

    _cache: dict = {}

    def _generate_and_load_golden(self, model_config, ds_name):
        if ds_name not in self._cache:
            self._cache[ds_name] = _run_model_generation(model_config, ds_name)
        return self._cache[ds_name]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_golden_path(model_config, ds_name):
    ds_cfg = model_config['model_datasets'][ds_name]
    return Path(ds_cfg['golden_npz']).expanduser()


def _run_model_generation(model_config, ds_name):
    """Run prepare_model_multi_reg with fixed parameters, return (wv, model)."""
    import yaml as _yaml
    from starships import retrieval as ret

    ds_cfg = model_config['model_datasets'][ds_name]

    # Check golden exists
    golden_path = _get_golden_path(model_config, ds_name)
    if not golden_path.exists():
        pytest.skip(f"Golden NPZ not found: {golden_path}")

    # Load the retrieval config YAML
    ret_cfg_path = Path(ds_cfg['retrieval_config']).expanduser()
    if not ret_cfg_path.exists():
        pytest.skip(f"Retrieval config not found: {ret_cfg_path}")

    with open(ret_cfg_path) as f:
        input_params = _yaml.safe_load(f)

    print(f"\n  [{ds_name}] Setting up retrieval from {ret_cfg_path.name} ...")
    ret.setup_retrieval(input_params)

    # Build theta array in params_prior order from stored parameter values
    # theta_params stores values in sampled space:
    #   log_uniform params → stored in log10 (will be exponentiated by unpack_theta)
    #   uniform params → stored at face value
    theta_params = ds_cfg['theta_params']
    missing = [k for k in ret.params_prior.keys() if k not in theta_params]
    if missing:
        pytest.skip(
            f"[{ds_name}] theta_params missing keys: {missing}. "
            "Re-generate golden with matching retrieval config."
        )

    theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

    print(f"  [{ds_name}] Generating model spectrum ...")
    theta_regions = ret.unpack_theta(theta)
    wv, model = ret.prepare_model_multi_reg(theta_regions, mode=ds_cfg['mode'])

    return wv, model
