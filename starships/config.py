"""
STARSHIPS global configuration.

Settings are stored in ~/.starships/config.yaml and can be edited with
edit_config() or directly in the file.

Usage::

    from starships.config import edit_config, get_output_dir

    # Set paths (run once on each machine)
    edit_config(
        data_dir='/scratch/user/starships_data',
        output_dir='/scratch/user/starships_output',
        scratch_dir='/scratch/user/starships_scratch',
        prt_input_data_path='/path/to/petitRADTRANS/input_data',
    )

    # Read paths in scripts
    out = get_output_dir() / 'wasp33b' / 'retrieval_run_001'
    out.mkdir(parents=True, exist_ok=True)
"""

from pathlib import Path
from typing import Optional
import yaml

_CONFIG_DIR  = Path.home() / '.starships'
_CONFIG_FILE = _CONFIG_DIR / 'config.yaml'

_DEFAULTS = {
    # Root directory for raw and reduced spectroscopic data
    'data_dir': None,
    # Root directory for final outputs (retrievals, analysis results)
    'output_dir': None,
    # Directory for temporary / intermediate files (e.g. HPC scratch)
    'scratch_dir': None,
    # petitRADTRANS input_data folder (corr-k and lbl opacity grids)
    'prt_input_data_path': None,
    # Golden outputs for regression tests (defaults to ~/.starships/regression_golden/)
    'regression_golden_dir': None,
}


# ---------------------------------------------------------------------------
# Core read/write
# ---------------------------------------------------------------------------

def _load() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    if not _CONFIG_FILE.exists():
        return _DEFAULTS.copy()
    with open(_CONFIG_FILE) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _DEFAULTS.copy()
    cfg.update(user_cfg)
    return cfg


def get_config() -> dict:
    """Return the full STARSHIPS configuration as a dict."""
    return _load()


def edit_config(**kwargs) -> None:
    """Update ~/.starships/config.yaml with new key-value pairs.

    Creates the file if it does not exist.

    Example::

        from starships.config import edit_config
        edit_config(output_dir='/scratch/user/starships_output')
        edit_config(prt_input_data_path='/path/to/petitRADTRANS/input_data')
    """
    unknown = set(kwargs) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}. "
                         f"Valid keys: {list(_DEFAULTS)}")
    _CONFIG_DIR.mkdir(exist_ok=True)
    cfg = _load()
    cfg.update(kwargs)
    with open(_CONFIG_FILE, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
    print(f"Configuration saved to {_CONFIG_FILE}")


def show_config() -> None:
    """Print the current configuration."""
    cfg = _load()
    print(f"STARSHIPS configuration  ({_CONFIG_FILE})")
    print("-" * 50)
    for key, val in sorted(cfg.items()):
        status = "" if val else "  ← not set"
        print(f"  {key:<28} {val}{status}")


# ---------------------------------------------------------------------------
# Convenience accessors  (return Path or None)
# ---------------------------------------------------------------------------

def _as_path(key: str, default: Optional[str] = None) -> Optional[Path]:
    val = _load().get(key) or default
    return Path(val) if val else None


def get_data_dir() -> Optional[Path]:
    """Root directory for raw and reduced spectroscopic data."""
    return _as_path('data_dir')


def get_output_dir() -> Optional[Path]:
    """Root directory for final outputs (retrievals, analysis results)."""
    return _as_path('output_dir')


def get_scratch_dir() -> Optional[Path]:
    """Directory for temporary / intermediate files."""
    return _as_path('scratch_dir')


def get_prt_input_data_path() -> Optional[Path]:
    """petitRADTRANS input_data folder."""
    return _as_path('prt_input_data_path')


def get_regression_golden_dir() -> Path:
    """Directory for regression test golden outputs.

    Defaults to ~/.starships/regression_golden/ if not explicitly set.
    """
    return _as_path(
        'regression_golden_dir',
        default=str(_CONFIG_DIR / 'regression_golden'),
    )
