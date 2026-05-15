"""
Generate golden (reference) outputs for STARSHIPS regression tests.

Run this ONCE on Narval with the current code before any cleanup or refactoring.
The outputs are saved in the directory returned by get_regression_golden_dir()
(defaults to ~/.starships/regression_golden/) and are used by the pytest
regression tests to verify that code changes do not alter scientific results.

Usage::

    # Default: all goldens (logl + model)
    python tests/regression/generate_golden.py

    # Custom config and output directory
    python tests/regression/generate_golden.py \\
        --config /path/to/regression_config.yaml \\
        --output-dir /path/to/golden/

    # Generate only specific type
    python tests/regression/generate_golden.py --only logl
    python tests/regression/generate_golden.py --only model
    python tests/regression/generate_golden.py --only reduction

    # Also save diagnostic plots (PNG) alongside the golden files
    python tests/regression/generate_golden.py --plots

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
    data = np.load(Path(ds_config['npz_path']).expanduser(), allow_pickle=True)
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


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_logl(ds_name, corrRV, logl_per_rv, logl_per_ord, plots_dir):
    """Save logL(RV) profile and logL per order as PNG files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # logL(RV) profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(corrRV, logl_per_rv, lw=1.5)
    peak_rv = corrRV[np.ma.argmax(logl_per_rv)]
    ax.axvline(peak_rv, color='r', ls='--', label=f'Peak = {peak_rv:+.1f} km/s')
    ax.set_xlabel('RV (km/s)')
    ax.set_ylabel('logL')
    ax.set_title(f'{ds_name} — logL(RV) profile')
    ax.legend()
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_logl_profile.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")

    # logL per order
    fig, ax = plt.subplots(figsize=(10, 4))
    orders = np.arange(len(logl_per_ord))
    ax.bar(orders, logl_per_ord, width=0.8)
    ax.set_xlabel('Order index')
    ax.set_ylabel('logL (summed over RV and exposures)')
    ax.set_title(f'{ds_name} — logL per order')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_logl_per_order.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")


def _plot_model(ds_name, wv, model, plots_dir):
    """Save model spectrum as a PNG file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wv, model, lw=0.8)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Model spectrum')
    ax.set_title(f'{ds_name} — model spectrum')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_model_spectrum.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")


def _plot_reduction(ds_name, transit, plots_dir):
    """Save PCA components and a few example spectra as PNG files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # PCA components
    comps = transit.pca.components_
    n_comp = min(5, comps.shape[0])
    fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(comps[i], lw=0.8)
        ax.set_ylabel(f'PC {i+1}')
    axes[-1].set_xlabel('Pixel index (flattened)')
    fig.suptitle(f'{ds_name} — PCA components')
    fig.tight_layout()
    out = plots_dir / f'{ds_name}_pca_components.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out}")

    # Example spectra: first order, first 5 exposures
    if transit.flux is not None:
        flux_data = np.ma.filled(transit.flux, np.nan)
        n_exp = min(5, flux_data.shape[0])
        wave = transit.wave  # shape (n_ord, n_pix) or similar
        fig, ax = plt.subplots(figsize=(12, 4))
        for i in range(n_exp):
            ax.plot(flux_data[i, 0, :], lw=0.6, alpha=0.7, label=f'exp {i}')
        ax.set_xlabel('Pixel index')
        ax.set_ylabel('Flux (order 0)')
        ax.set_title(f'{ds_name} — first {n_exp} exposures, order 0')
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = plots_dir / f'{ds_name}_flux_example.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Plot : {out}")


# ---------------------------------------------------------------------------
# Golden generators
# ---------------------------------------------------------------------------

def generate_logl_goldens(cfg, output_dir, plots_dir=None):
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

        model    = np.load(Path(ds_cfg['model_path']).expanduser())
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

        if plots_dir is not None:
            _plot_logl(ds_name, corrRV, logl_per_rv, logl_per_ord, plots_dir)

    return summary


def generate_model_goldens(cfg, output_dir, plots_dir=None):
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

        print(f"  Setting up retrieval from {ret_cfg_path.name} ...")
        ret.setup_retrieval(input_parameters=ret_cfg_path)

        theta_params = ds_cfg['theta_params']
        missing = [k for k in ret.params_prior.keys() if k not in theta_params]
        if missing:
            print(f"  SKIP: theta_params missing keys: {missing}")
            continue

        theta = np.array([theta_params[k] for k in ret.params_prior.keys()])

        print(f"  Generating model spectrum (mode={ds_cfg['mode']}) ...")
        theta_regions = ret.unpack_theta(theta)
        wv, model = ret.prepare_model_multi_reg(theta_regions, mode=ds_cfg['mode'])

        out_path = Path(ds_cfg['golden_npz']).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, wv=wv, model=model)

        print(f"  wv    : {wv.shape}  [{wv.min():.4f}, {wv.max():.4f}] µm")
        print(f"  model : {model.shape}  [{model.min():.3e}, {model.max():.3e}]")
        print(f"  Saved : {out_path}")
        summary[ds_name] = {'wv_range': [float(wv.min()), float(wv.max())],
                            'model_max': float(model.max())}

        if plots_dir is not None:
            _plot_model(ds_name, wv, model, plots_dir)

    return summary


def generate_reduction_goldens(cfg, plots_dir=None):
    """Re-run the reduction pipeline and overwrite the golden NPZ files.

    The golden_npz path in reduction_datasets IS the reference file — running
    this function regenerates it on the current machine so that regression tests
    pass on the same platform (MKL on Narval, Accelerate on macOS, etc.).
    """
    import yaml as _yaml
    import numpy as np
    import starships.planet_obs as pl_obs
    from starships.planet_obs import Observations
    from pipeline.reduction import pl_param_units

    summary = {}
    for ds_name, ds_cfg in cfg.get('reduction_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}  (reduction)")
        print(f"{'='*60}")

        pipeline_cfg_path = Path(ds_cfg['pipeline_config']).expanduser()
        if not pipeline_cfg_path.exists():
            print(f"  SKIP: pipeline config not found: {pipeline_cfg_path}")
            continue

        with open(pipeline_cfg_path) as f:
            config_dict = _yaml.safe_load(f)

        obs_dir = Path(config_dict.get('obs_dir', '')).expanduser()
        if not obs_dir.exists():
            print(f"  SKIP: raw data directory not found: {obs_dir}")
            continue

        golden_path = Path(ds_cfg['golden_npz']).expanduser()

        pl_kwargs = pl_param_units(config_dict) if config_dict.get('pl_params') else {}

        visit_name = ds_cfg['visit_name']
        list_filenames = {
            'list_e2ds':  f'list_e2ds_{visit_name}',
            'list_tcorr': f'list_tcorr_{visit_name}',
            'list_recon': f'list_recon_{visit_name}',
        }

        print(f"  Loading raw data from {obs_dir} (visit: {visit_name}) ...")
        obs = Observations(name=config_dict['pl_name'], pl_kwargs=pl_kwargs)
        obs.fetch_data(obs_dir, **list_filenames)
        obs.n_spec = len(obs.filenames)

        all_exp = np.arange(obs.n_spec)
        bad = config_dict.get('bad_indexs', {}).get(visit_name, [])
        transit_tags = [np.delete(all_exp, bad) if bad else all_exp]

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

        print(f"  Running reduction "
              f"(n_pc={n_pc}, mask_tellu={mask_tellu}, mask_wings={mask_wings}) ...")
        list_tr = pl_obs.generate_all_transits(
            obs, transit_tags, [0.0], params_all, config_dict['iout_all'],
            counting=False, **kwargs_gen_tr, **kwargs_build_ts,
        )
        transit = list_tr['1']

        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            golden_path,
            flux         = transit.flux.data,
            mask_flux    = transit.flux.mask,
            wave         = transit.wave,
            noise        = transit.noise.data,
            components_  = transit.pca.components_,
        )
        print(f"  flux  : {transit.flux.shape}")
        print(f"  wave  : {transit.wave.shape}")
        print(f"  Saved : {golden_path}")
        summary[ds_name] = {'n_spec': transit.flux.shape[0],
                            'n_ord':  transit.flux.shape[1]}

        if plots_dir is not None:
            _plot_reduction(ds_name, transit, plots_dir)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        help='Override output directory for logL/model golden files '
             '(default: from starships.config.get_regression_golden_dir())',
    )
    parser.add_argument(
        '--only',
        choices=['logl', 'model', 'reduction'],
        default=None,
        help='Generate only one type of golden (default: logl + model)',
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Save diagnostic PNG plots alongside the golden files',
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

    plots_dir = (output_dir / 'plots') if args.plots else None

    summary = {}

    if args.only == 'logl' or args.only is None:
        logl_summary = generate_logl_goldens(cfg, output_dir, plots_dir=plots_dir)
        summary.update(logl_summary)

    if args.only == 'model' or args.only is None:
        model_summary = generate_model_goldens(cfg, output_dir, plots_dir=plots_dir)
        summary.update({f'[model] {k}': v for k, v in model_summary.items()})

    if args.only == 'reduction':
        red_summary = generate_reduction_goldens(cfg, plots_dir=plots_dir)
        summary.update({f'[reduction] {k}': v for k, v in red_summary.items()})

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, res in summary.items():
        if 'peak_rv' in res:
            print(f"  {name:<40}  peak RV = {res['peak_rv']:+7.1f} km/s  "
                  f"logL_max = {res['max_logl']:.4f}")
        elif 'wv_range' in res:
            print(f"  {name:<40}  wv [{res['wv_range'][0]:.3f}, "
                  f"{res['wv_range'][1]:.3f}] µm  model_max = {res['model_max']:.3e}")
        else:
            print(f"  {name:<40}  {res['n_spec']} exposures, {res['n_ord']} orders")

    print(f"\n  Golden outputs saved to: {output_dir.resolve()}")
    if plots_dir is not None:
        print(f"  Plots saved to:          {plots_dir.resolve()}")
    print("\n  Next: after any code change, run")
    print("    pytest tests/regression/ -v")


if __name__ == '__main__':
    main()
