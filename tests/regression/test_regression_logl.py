"""
Regression tests for logL computations.

These tests verify that numerical results are unchanged after non-scientific
code modifications (cleanup, refactoring, dependency updates, etc.).

Prerequisites (on Narval):
    1. Set paths in ~/.starships/config.yaml (via starships.config.edit_config)
    2. Copy config_template.yaml → ~/.starships/regression_config.yaml and fill in paths
    3. Run generate_golden.py once to create golden references
    4. Run `pytest tests/regression/ -v` after every code change

Tests are automatically SKIPPED if the config or golden outputs are missing,
so they do not break local development or CI environments without real data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

_REGRESSION_CONFIG = Path.home() / '.starships' / 'regression_config.yaml'

_config_available = _REGRESSION_CONFIG.exists()


def _golden_dir():
    from starships.config import get_regression_golden_dir
    return get_regression_golden_dir()


def _dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('datasets', {}).keys())


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_regression_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "Regression config not found. "
        "On Narval: copy config_template.yaml → ~/.starships/regression_config.yaml"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def regression_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='session')
def corrRV(regression_config):
    g = regression_config['rv_grid']
    return np.arange(g['min'], g['max'] + g['step'], g['step'])


def _load_wave_flux(ds_config):
    path = Path(ds_config['npz_path']).expanduser()
    if not path.exists():
        pytest.skip(f"Reduced data not found (configure on Narval or locally): {path}")
    data = np.load(path, allow_pickle=True)
    wave = data['wave']
    mask = data.get('mask_flux', np.zeros(data['flux'].shape, dtype=bool))
    return wave, np.ma.array(data['flux'], mask=mask)


def _load_golden(ds_name):
    path = _golden_dir() / f'{ds_name}_logl.npz'
    if not path.exists():
        pytest.skip(f"Golden output missing for {ds_name}: {path}\n"
                    "Run generate_golden.py to create it.")
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogLProfileClassic:
    """
    Verify that quick_correl (Classic_correlation approach) produces identical
    numerical results before and after code modifications.

    Tolerance: rtol=1e-10 for full profile, rtol=1e-12 for spot checks.
    These are deterministic floating-point operations on the same machine.
    """

    @requires_regression_data
    @pytest.mark.parametrize("ds_name", _dataset_names())
    def test_logl_profile_unchanged(self, regression_config, corrRV, ds_name):
        """Full logL(RV) profile must match the golden reference."""
        from starships.correlation import quick_correl

        ds_cfg   = regression_config['datasets'][ds_name]
        wave, flux = _load_wave_flux(ds_cfg)
        model    = np.load(Path(ds_cfg['model_path']).expanduser())

        correl = quick_correl(wave, flux, corrRV,
                              model['wave'], model['spec'],
                              get_logl=True, kind='BL', counting=False)
        logl_per_rv = np.ma.sum(correl, axis=(0, 1))

        golden   = _load_golden(ds_name)
        ref      = np.ma.array(golden['logl_per_rv'],
                               mask=golden['logl_per_rv_mask'])

        np.testing.assert_allclose(
            logl_per_rv.data, ref.data, rtol=1e-10,
            err_msg=f"[{ds_name}] logL(RV) profile differs from golden reference",
        )

    @requires_regression_data
    @pytest.mark.parametrize("ds_name", _dataset_names())
    def test_spot_checks_exact(self, regression_config, corrRV, ds_name):
        """A small set of individual correl values must match bit-for-bit."""
        from starships.correlation import quick_correl

        ds_cfg   = regression_config['datasets'][ds_name]
        wave, flux = _load_wave_flux(ds_cfg)
        model    = np.load(Path(ds_cfg['model_path']).expanduser())

        correl = quick_correl(wave, flux, corrRV,
                              model['wave'], model['spec'],
                              get_logl=True, kind='BL', counting=False)

        golden = _load_golden(ds_name)
        si, sj, sk = golden['spot_shape']
        np.testing.assert_allclose(
            np.array(correl[:si, :sj, :sk]),
            golden['spot_checks'],
            rtol=1e-12,
            err_msg=f"[{ds_name}] Spot-check values differ from golden reference",
        )

    @requires_regression_data
    @pytest.mark.parametrize("ds_name", _dataset_names())
    def test_peak_rv_unchanged(self, regression_config, corrRV, ds_name):
        """The logL peak position (RV) must not shift after code changes."""
        from starships.correlation import quick_correl

        ds_cfg   = regression_config['datasets'][ds_name]
        wave, flux = _load_wave_flux(ds_cfg)
        model    = np.load(Path(ds_cfg['model_path']).expanduser())

        correl = quick_correl(wave, flux, corrRV,
                              model['wave'], model['spec'],
                              get_logl=True, kind='BL', counting=False)
        logl_per_rv = np.ma.sum(correl, axis=(0, 1))
        peak_rv = corrRV[np.ma.argmax(logl_per_rv)]

        golden      = _load_golden(ds_name)
        peak_rv_ref = golden['corrRV'][np.argmax(golden['logl_per_rv'])]
        rv_step     = regression_config['rv_grid']['step']

        assert abs(peak_rv - peak_rv_ref) <= rv_step / 2, (
            f"[{ds_name}] logL peak shifted: {peak_rv:.1f} km/s "
            f"(was {peak_rv_ref:.1f} km/s)"
        )

        # Optional: check against known astrophysical value
        expected     = ds_cfg.get('expected_peak_rv')
        expected_tol = ds_cfg.get('expected_peak_rv_tol', rv_step * 2)
        if expected is not None:
            assert abs(peak_rv - expected) <= expected_tol, (
                f"[{ds_name}] Peak RV unexpected: {peak_rv:.1f} km/s "
                f"(expected {expected:.1f} ± {expected_tol:.1f} km/s)"
            )
