"""
Generate golden (reference) outputs for STARSHIPS regression tests.

Run this ONCE on Narval with the current code before any cleanup or refactoring.
The outputs are saved in the directory returned by get_regression_golden_dir()
(defaults to ~/.starships/regression_golden/) and are used by the pytest
regression tests to verify that code changes do not alter scientific results.

Usage::

    # Default: reads ~/.starships/regression_config.yaml
    python tests/regression/generate_golden.py

    # Custom config and output directory
    python tests/regression/generate_golden.py \\
        --config /path/to/regression_config.yaml \\
        --output-dir /path/to/golden/

Setup::

    from starships.config import edit_config
    edit_config(regression_golden_dir='/scratch/user/starships_regression/golden')
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import yaml


def load_regression_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_wave_flux(ds_config: dict):
    """Load wave and masked flux from a reduced NPZ file."""
    data = np.load(ds_config['npz_path'], allow_pickle=True)
    wave = data['wave']
    mask = data.get('mask_flux', np.zeros(data['flux'].shape, dtype=bool))
    flux = np.ma.array(data['flux'], mask=mask)
    return wave, flux


def compute_logl_profile(wave, flux, wave_mod, spec_mod, corrRV):
    """Compute logL profile over RV grid using quick_correl (Brogi & Line).

    Returns
    -------
    correl       : masked array (n_spec, n_ord, n_rv)
    logl_per_rv  : masked array (n_rv,)  — summed over exposures and orders
    logl_per_ord : masked array (n_ord,) — summed over exposures and RVs
    """
    from starships.correlation import quick_correl

    correl = quick_correl(wave, flux, corrRV, wave_mod, spec_mod,
                          get_logl=True, kind='BL', counting=False)
    logl_per_rv  = np.ma.sum(correl, axis=(0, 1))
    logl_per_ord = np.ma.sum(correl, axis=(0, 2))
    return correl, logl_per_rv, logl_per_ord


def generate_logl_goldens(cfg, output_dir):
    """Generate golden NPZ files for logL regression tests."""
    corrRV = np.arange(
        cfg['rv_grid']['min'],
        cfg['rv_grid']['max'] + cfg['rv_grid']['step'],
        cfg['rv_grid']['step'],
    )

    summary = {}
    for ds_name, ds_cfg in cfg.get('datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        wave, flux = load_wave_flux(ds_cfg)
        print(f"  wave  : {wave.shape}")
        print(f"  flux  : {flux.shape}  (masked: {int(flux.mask.sum())})")

        model    = np.load(ds_cfg['model_path'])
        wave_mod = model['wave']
        spec_mod = model['spec']
        print(f"  model : {wave_mod.shape[0]} pts, "
              f"[{wave_mod.min():.4f}, {wave_mod.max():.4f}] µm")

        correl, logl_per_rv, logl_per_ord = compute_logl_profile(
            wave, flux, wave_mod, spec_mod, corrRV
        )

        n_spec, n_ord, _ = correl.shape
        si, sj, sk = min(3, n_spec), min(3, n_ord), min(5, len(corrRV))
        spot_checks = np.array(correl[:si, :sj, :sk])

        out_path = output_dir / f'{ds_name}_logl.npz'
        np.savez(
            out_path,
            corrRV        = corrRV,
            logl_per_rv   = logl_per_rv.data,
            logl_per_rv_mask = (logl_per_rv.mask
                                if hasattr(logl_per_rv, 'mask')
                                else np.zeros(len(logl_per_rv), dtype=bool)),
            logl_per_ord  = logl_per_ord.data,
            spot_checks   = spot_checks,
            spot_shape    = np.array([si, sj, sk]),
        )

        peak_rv = corrRV[np.ma.argmax(logl_per_rv)]
        print(f"  Peak  : RV = {peak_rv:+.1f} km/s")
        print(f"  Max logL : {float(logl_per_rv.max()):.6f}")
        print(f"  Saved : {out_path}")
        summary[ds_name] = {'peak_rv': float(peak_rv),
                            'max_logl': float(logl_per_rv.max())}
    return summary


def generate_model_goldens(cfg, output_dir):
    """Generate golden NPZ files for model regression tests (requires petitRADTRANS)."""
    try:
        import petitRADTRANS  # noqa: F401
    except ImportError:
        print("\n  [model] petitRADTRANS not available — skipping model goldens.")
        return {}

    from starships import retrieval as ret

    summary = {}
    for ds_name, ds_cfg in cfg.get('model_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}  (model)")
        print(f"{'='*60}")

        ret_cfg_path = Path(ds_cfg['retrieval_config']).expanduser()
        if not ret_cfg_path.exists():
            print(f"  SKIP: retrieval config not found: {ret_cfg_path}")
            continue

        with open(ret_cfg_path) as f:
            input_params = yaml.safe_load(f)

        print(f"  Setting up retrieval from {ret_cfg_path.name} ...")
        ret.setup_retrieval(input_params)

        theta_params = ds_cfg['theta_params']
        missing = [k for k in ret.params_prior.keys() if k not in theta_params]
        if missing:
            print(f"  SKIP: theta_params missing keys: {missing}")
            continue

        theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

        print(f"  Generating model spectrum (mode={ds_cfg['mode']}) ...")
        theta_regions = ret.unpack_theta(theta)
        wv, model = ret.prepare_model_multi_reg(theta_regions, mode=ds_cfg['mode'])

        # Store golden: wv + model + the theta_params used (for traceability)
        out_path = Path(ds_cfg['golden_npz']).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            wv    = wv,
            model = model,
        )

        print(f"  wv    : {wv.shape}  [{wv.min():.4f}, {wv.max():.4f}] µm")
        print(f"  model : {model.shape}  [{model.min():.3e}, {model.max():.3e}]")
        print(f"  Saved : {out_path}")
        summary[ds_name] = {'wv_range': [float(wv.min()), float(wv.max())],
                            'model_max': float(model.max())}
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate regression golden outputs for STARSHIPS'
    )
    parser.add_argument(
        '--config',
        default=str(Path.home() / '.starships' / 'regression_config.yaml'),
        help='Path to regression config YAML (default: ~/.starships/regression_config.yaml)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Override output directory for golden files '
             '(default: from starships.config.get_regression_golden_dir())',
    )
    parser.add_argument(
        '--only',
        choices=['logl', 'model'],
        default=None,
        help='Generate only logL or model goldens (default: both)',
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        print("Copy tests/regression/config_template.yaml to "
              "~/.starships/regression_config.yaml and fill in the paths.")
        sys.exit(1)

    from starships.config import get_regression_golden_dir

    cfg = load_regression_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else get_regression_golden_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if args.only != 'model':
        logl_summary = generate_logl_goldens(cfg, output_dir)
        summary.update(logl_summary)

    if args.only != 'logl':
        model_summary = generate_model_goldens(cfg, output_dir)
        summary.update({f'[model] {k}': v for k, v in model_summary.items()})

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, res in summary.items():
        if 'peak_rv' in res:
            print(f"  {name:<36}  peak RV = {res['peak_rv']:+7.1f} km/s  "
                  f"logL_max = {res['max_logl']:.4f}")
        else:
            print(f"  {name:<36}  wv [{res['wv_range'][0]:.3f}, "
                  f"{res['wv_range'][1]:.3f}] µm  model_max = {res['model_max']:.3e}")

    print(f"\n  Golden outputs saved to: {output_dir.resolve()}")
    print("\n  Next: after any code change, run")
    print("    pytest tests/regression/ -v")


if __name__ == '__main__':
    main()
