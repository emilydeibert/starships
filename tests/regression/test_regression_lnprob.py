"""
Regression tests for the lnprob (log-probability) function in the retrieval pipeline.

These tests run the full retrieval chain end-to-end:
    setup_retrieval → load_high_res_data → lnprob(theta)

and verify that the scalar result (prior + logL) is numerically unchanged
after code modifications.  This is the most comprehensive regression test:
it exercises the pRT model generation, the orbital-motion correction, the
PCA subtraction, and the logL prescription all at once.

Requirements:
    - petitRADTRANS installed and opacity tables accessible (Narval only)
    - Reduced NPZ high-res data accessible (paths set in the retrieval YAML)
    - ~/.starships/regression_config.yaml with a 'lnprob_datasets' section

These tests are opt-in only.  Run with:

    pytest tests/regression/ -v -m lnprob

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


def _lnprob_dataset_names():
    if not _config_available:
        return []
    with open(_REGRESSION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get('lnprob_datasets', {}).keys())


pytestmark = pytest.mark.lnprob  # entire module requires -m lnprob


requires_lnprob_data = pytest.mark.skipif(
    not _config_available,
    reason=(
        "lnprob config not found. "
        "On Narval: add 'lnprob_datasets' to ~/.starships/regression_config.yaml"
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
def lnprob_config():
    with open(_REGRESSION_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLnprob:
    """
    Run lnprob(theta) end-to-end and verify the scalar result is numerically
    identical to the golden reference.

    The test exercises the complete chain:
      pRT model generation → orbital-motion RV shift → PCA subtraction → logL

    Tolerance: rtol=1e-10 (deterministic floating-point on the same machine).
    """

    _cache: dict = {}

    def _lnprob_value(self, lnprob_config, ds_name):
        """Evaluate lnprob once per dataset and cache the result."""
        if ds_name not in self._cache:
            ds_cfg = lnprob_config['lnprob_datasets'][ds_name]
            self._cache[ds_name] = _run_lnprob(ds_cfg)
        return self._cache[ds_name]

    @requires_lnprob_data
    @requires_prt
    @pytest.mark.parametrize("ds_name", _lnprob_dataset_names())
    def test_lnprob_unchanged(self, lnprob_config, ds_name):
        """lnprob(theta) must match the golden reference (rtol=1e-10)."""
        lnprob_val = self._lnprob_value(lnprob_config, ds_name)
        golden     = _load_golden(ds_name)

        np.testing.assert_allclose(
            lnprob_val, float(golden['lnprob']),
            rtol=1e-10,
            err_msg=f"[{ds_name}] lnprob differs from golden reference",
        )

    @requires_lnprob_data
    @requires_prt
    @pytest.mark.parametrize("ds_name", _lnprob_dataset_names())
    def test_lnprob_is_finite(self, lnprob_config, ds_name):
        """lnprob(theta) must be finite at the reference point.

        A -inf result means theta_params is outside the prior or the model
        produced NaNs — the test data or prior bounds in the config may be stale.
        """
        lnprob_val = self._lnprob_value(lnprob_config, ds_name)
        assert np.isfinite(lnprob_val), (
            f"[{ds_name}] lnprob = {lnprob_val} is not finite. "
            "Check that theta_params are within the prior bounds "
            "and that the retrieval config matches the current code."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _golden_dir():
    from starships.config import get_regression_golden_dir
    return get_regression_golden_dir()


def _load_golden(ds_name):
    path = _golden_dir() / f'{ds_name}_lnprob.npz'
    if not path.exists():
        pytest.skip(
            f"Golden output missing for {ds_name}: {path}\n"
            "Run: python tests/regression/generate_golden.py --only lnprob"
        )
    return np.load(path, allow_pickle=True)


def _run_lnprob(ds_cfg):
    """Set up the retrieval, load high-res data, and evaluate lnprob at theta."""
    from starships import retrieval as ret

    ret_cfg_path = Path(ds_cfg['retrieval_config']).expanduser()
    if not ret_cfg_path.exists():
        pytest.skip(f"Retrieval config not found: {ret_cfg_path}")

    print(f"\n  Setting up retrieval from {ret_cfg_path.name} ...")
    ret.setup_retrieval(input_parameters=ret_cfg_path)

    print("  Loading high-res data ...")
    ret.load_high_res_data()

    theta_params = ds_cfg['theta_params']
    missing = [k for k in ret.params_prior.keys() if k not in theta_params]
    if missing:
        pytest.skip(
            f"theta_params missing keys: {missing}. "
            "Update theta_params in lnprob_datasets to match the retrieval config."
        )

    theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

    print(f"  Evaluating lnprob (n_params={len(theta)}) ...")
    return float(ret.lnprob(theta))
