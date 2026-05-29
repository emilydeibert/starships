"""
logl_grid — Kp × v_sys logL grid computation for high-resolution spectroscopy.

Workflow
--------
1. Compute::

    setup_logl_grid(yaml) → load_logl_grid_data() → load_model(npz)
    → results = compute_logl_grid() → save_logl_grid(results, out_path, stem)

2. Analyse::

    load_logl_results(file_list, path)
    logl = get_logl(alpha=1., sum_axis=(-2, -1))

The module uses module-level globals (same pattern as starships.retrieval) so
that multiprocessing workers inherit large arrays without memory duplication via
fork.

Extra keys to add to the retrieval YAML
----------------------------------------
::

    rv_grid:  {min: -200, max: 200,  step: 5}   # km/s
    kp_grid:  {min: 0,   max: 500,  step: 5}    # km/s
    logl_kind: BL                                # BL or G
    logl_grid_output_path: ~/scratch/DataAnalysis/SPIRou/logl_grids
    n_processes_per_cpu: 3
    apply_alpha: true                            # true = modulate template by alpha_frac (realistic);
                                                 # false = uniform injection (all frames equal weight)

Output filenames encode the grid parameters so that different grids for the
same model do not overwrite each other::

    {model_stem}_kp{min}_{max}_{step}_rv{min}_{max}_{step}[_noalpha]_visit{i}.npz

Negative values are written with a leading ``m`` (e.g., -100 → m100).

Command-line usage (Narval)
---------------------------
::

    python -m starships.logl_grid \
        --config ~/scratch/.../retrieval_inputs.yaml \
        --specfile take3_HRR_all.npz
"""

import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy import units as u

import starships.planet_obs as pl_obs
from starships import correlation as corr
from starships.orbite import rv_theo_t
from starships.planet_obs import Observations

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level globals
# Declared here for discoverability; set by setup_logl_grid / load_* functions.
# ---------------------------------------------------------------------------

# --- From setup_logl_grid (YAML keys) ---
global pl_name, pl_params
global high_res_file_stem_list, high_res_path
global instrum, instrum_param_list, wv_range_high
global kind_trans, orders
# logl_grid-specific
global rv_grid, kp_grid, logl_kind
global logl_grid_output_path, n_processes_per_cpu
global apply_alpha
# Derived in setup_logl_grid
global obs, planet, Kp_scale

# --- From load_logl_grid_data ---
global data_trs, data_info_list, idx_orders, axis_sum

# --- From load_model ---
global wv_high, model_high

# --- Workers (set per-visit before Pool.map) ---
global _current_data_tr, _current_alpha_frac


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# NOTE: This function must live in *this* module so that globals() modifies
# logl_grid's own namespace (same trick as retrieval.setup_retrieval).
def setup_logl_grid(input_parameters, **kwargs):
    """Set up module globals for logl grid computation.

    Reads the same YAML as ``setup_retrieval``, plus the logl_grid-specific
    keys listed in the module docstring.
    """
    from starships.retrieval import unpack_input_parameters, get_wv_range
    from starships.instruments import load_instrum

    input_params = unpack_input_parameters(input_parameters, **kwargs)

    _KEYS = [
        'pl_name', 'pl_params',
        'high_res_file_stem_list', 'high_res_path',
        'instrum', 'kind_trans', 'orders',
        'rv_grid', 'kp_grid', 'logl_kind',
        'logl_grid_output_path', 'n_processes_per_cpu',
        'apply_alpha',
    ]
    for key in _KEYS:
        if key in input_params:
            globals()[key] = input_params[key]

    # Defaults for optional logl_grid keys
    global rv_grid, kp_grid, logl_kind, n_processes_per_cpu, apply_alpha
    if 'rv_grid' not in globals():
        rv_grid = None
    if 'kp_grid' not in globals():
        kp_grid = None
    logl_kind = input_params.get('logl_kind', 'BL')
    n_processes_per_cpu = input_params.get('n_processes_per_cpu', 3)
    apply_alpha = bool(input_params.get('apply_alpha', True))

    # Build planet / obs
    global obs, planet, Kp_scale
    obs = Observations(name=input_params['pl_name'], pl_kwargs=input_params['pl_params'])
    planet = obs.planet
    Kp_scale = (planet.M_pl / planet.M_star).decompose().value
    planet.all_params = None  # free unused memory

    # Instrument info
    global instrum_param_list, wv_range_high
    instrum_param_list = [load_instrum(name) for name in input_params['instrum']]
    wv_range_high = get_wv_range([p['high_res_wv_lim'] for p in instrum_param_list])
    log.info(f'Wavelength range (high res): {wv_range_high}')


def load_logl_grid_data():
    """Load high-res data into globals. Run after setup_logl_grid."""
    global data_trs, data_info_list, idx_orders, axis_sum

    data_trs = []
    data_info_list = []

    for high_res_file_stem in high_res_file_stem_list:
        log.info(f'Loading: {high_res_path / high_res_file_stem}')
        data_info_i, data_trs_i = pl_obs.load_sequences(
            high_res_file_stem, [1], path=high_res_path
        )
        data_trs.append(data_trs_i['0'])
        data_info_list.append(data_info_i)

    n_ord = data_trs[0]['flux'].shape[1]
    idx_orders = orders if orders is not None else np.arange(n_ord)
    axis_sum = -1  # sum over spectral pixel axis


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path):
    """Load a model spectrum NPZ. Sets ``wv_high`` and ``model_high`` globals."""
    global wv_high, model_high
    model_file = np.load(model_path)
    wv_high = model_file['wave']
    model_high = model_file['spec']
    model_file.close()
    if not np.isfinite(model_high[100:-100]).all():
        raise ValueError(f'NaN found in model spectrum: {model_path}')
    log.info(f'Model loaded: {Path(model_path).name}')


# ---------------------------------------------------------------------------
# Chi2 terms (per-exposure, per-order)
# ---------------------------------------------------------------------------

def _calc_chi2_terms(model, data_tr):
    """Compute model-dependent chi2 terms (f×g, g²) for one visit."""
    flux = data_tr['flux'][:, idx_orders]
    noise = data_tr['noise'][:, idx_orders]
    model_norm = model[:, idx_orders] / noise
    f_x_g = np.ma.sum(model_norm * flux, axis=axis_sum)
    s2g = np.ma.sum(model_norm ** 2, axis=axis_sum)
    return f_x_g, s2g


# Set per-visit in compute_logl_grid before Pool.map; inherited by workers via fork.
_current_data_tr = None
_current_alpha_frac = None
_current_apply_alpha = True


def _get_chi2_detailed(theta):
    """Worker: compute chi2 terms for one (v_sys, kp) grid point."""
    v_sys, kp = theta
    data_tr = _current_data_tr

    vrp_orb = rv_theo_t(
        kp, data_tr['t_start'] * u.d, planet.mid_tr, planet.period, plnt=True
    ).value

    n_pc = int(data_tr['params'][5])
    velocities = v_sys + vrp_orb - vrp_orb * Kp_scale + data_tr['RV_const']

    alpha_arg = _current_alpha_frac if _current_apply_alpha else np.ones_like(_current_alpha_frac)

    model_seq = corr.gen_model_sequence_noinj(
        velocities,
        data_wave=data_tr['wave'],
        data_sep=data_tr['sep'],
        data_pca=data_tr['pca'],
        data_npc=n_pc,
        planet=planet,
        model_wave=wv_high[20:-20],
        model_spec=model_high[20:-20],
        kind_trans=kind_trans,
        alpha=alpha_arg,
    )
    return _calc_chi2_terms(model_seq, data_tr)


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def _make_grid(grid_cfg):
    # step/2 as the upper-bound buffer: includes max without ever adding an extra point.
    # Using step itself as the buffer causes np.arange to overshoot when max is not an
    # exact multiple of step (e.g. min=227, max=228, step=5 would give [227, 232]).
    return np.arange(grid_cfg['min'], grid_cfg['max'] + grid_cfg['step'] / 2, grid_cfg['step'])


def compute_logl_grid(rv_array=None, kp_array=None, n_process=None):
    """Run Pool.map over (rv, kp) grid for all loaded visits.

    Returns
    -------
    list of dict
        One dict per visit with keys: cross_terms, squared_terms, kp, vsys.
    """
    global _current_data_tr, _current_alpha_frac, _current_apply_alpha

    if rv_array is None:
        rv_array = _make_grid(rv_grid)
    if kp_array is None:
        kp_array = _make_grid(kp_grid)
    if n_process is None:
        try:
            n_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            n_cpu = 1
        n_process = n_cpu * n_processes_per_cpu

    kp_step  = kp_array[1]  - kp_array[0]  if len(kp_array) > 1 else float('nan')
    rv_step  = rv_array[1]  - rv_array[0]  if len(rv_array) > 1 else float('nan')
    log.info(
        f'Kp  grid: [{kp_array[0]:.2f}, {kp_array[-1]:.2f}] km/s  '
        f'step={kp_step:.2f}  n={len(kp_array)}'
    )
    log.info(
        f'vsys grid: [{rv_array[0]:.2f}, {rv_array[-1]:.2f}] km/s  '
        f'step={rv_step:.2f}  n={len(rv_array)}'
    )
    log.info(
        f'Grid size: {len(rv_array) * len(kp_array)} points  '
        f'({len(rv_array)} × {len(kp_array)})'
    )
    log.info(f'apply_alpha = {apply_alpha}  (template modulated by alpha_frac: {apply_alpha})')

    _current_apply_alpha = apply_alpha

    kp_mesh, vsys_mesh = np.meshgrid(kp_array, rv_array)
    theta_grid = np.array([np.ravel(vsys_mesh), np.ravel(kp_mesh)]).T

    results = []
    for i, (data_tr, di) in enumerate(zip(data_trs, data_info_list), start=1):
        _current_data_tr = data_tr
        _current_alpha_frac = di['trall_alpha_frac']

        log.info(f'Computing grid for visit {i}/{len(data_trs)} with {n_process} processes ...')
        with Pool(n_process) as pool:
            outputs = pool.map(_get_chi2_detailed, theta_grid)

        outputs = np.array(outputs)
        data_shape = outputs.shape[2:]
        cross_terms, squared_terms = [
            np.reshape(outputs[:, idx], (*kp_mesh.shape, *data_shape))
            for idx in range(2)
        ]
        results.append(dict(
            cross_terms=cross_terms,
            squared_terms=squared_terms,
            kp=kp_mesh,
            vsys=vsys_mesh,
        ))
    return results


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def _fmt_v(v):
    """Format a grid value for use in a filename (no '.', 'm' for minus)."""
    v = float(v)
    iv = int(v)
    s = str(iv) if iv == v else f'{v:.3g}'.replace('.', 'p')
    return s.replace('-', 'm')


def _build_grid_stem_suffix(rv_array, kp_array, apply_alpha_flag):
    """Build the grid-parameter suffix for the output filename.

    Example: ``_kp200_300_5_rvm100_100_1`` or ``_kp200_300_5_rvm100_100_1_noalpha``
    """
    kp_step = kp_array[1] - kp_array[0] if len(kp_array) > 1 else 0
    rv_step = rv_array[1] - rv_array[0] if len(rv_array) > 1 else 0
    suffix = (
        f'_kp{_fmt_v(kp_array[0])}_{_fmt_v(kp_array[-1])}_{_fmt_v(kp_step)}'
        f'_rv{_fmt_v(rv_array[0])}_{_fmt_v(rv_array[-1])}_{_fmt_v(rv_step)}'
    )
    if not apply_alpha_flag:
        suffix += '_noalpha'
    return suffix


def _compute_contact_phases(planet_obj, kind_trans_str):
    """Return contact phases [T1, T2, T3, T4] in phase units.

    T1/T4 bracket the full transit/eclipse (external contacts).
    T2/T3 bracket the fully-in phase (internal contacts).

    Phases are relative to phase 0 for transmission and phase 0.5 for
    emission (secondary eclipse centred at 0.5).

    If the orbital geometry does not allow computing T2/T3 (grazing or
    missing parameters), T2 = T1 and T3 = T4.
    """
    # Force scalar floats: planet Quantity values can be 1-element arrays.
    T14_phase = float((planet_obj.trandur / planet_obj.period).decompose().value)

    phase_T1 = -T14_phase / 2
    phase_T4 = +T14_phase / 2
    phase_T2, phase_T3 = phase_T1, phase_T4  # fallback

    try:
        k = float((planet_obj.R_pl / planet_obj.R_star).decompose().value)
        # Impact parameter for a circular orbit: b = a·cos(i)/R_star
        b = float((planet_obj.ap * np.cos(planet_obj.incl) / planet_obj.R_star).decompose().value)

        denom = (1 + k) ** 2 - b ** 2
        numer = (1 - k) ** 2 - b ** 2

        if denom > 0 and numer > 0:
            # Winn (2010): T23/T14 = arcsin(sqrt(numer)) / arcsin(sqrt(denom))
            T23_phase = T14_phase * np.arcsin(np.sqrt(numer)) / np.arcsin(np.sqrt(denom))
            phase_T2 = -T23_phase / 2
            phase_T3 = +T23_phase / 2
    except Exception as exc:
        log.debug(f'Could not compute T23: {exc}')

    if kind_trans_str == 'emission':
        return np.array([0.5 + phase_T1, 0.5 + phase_T2, 0.5 + phase_T3, 0.5 + phase_T4])
    return np.array([phase_T1, phase_T2, phase_T3, phase_T4])


def save_logl_grid(results, output_path, file_stem):
    """Save grid results to NPZ, one file per visit.

    The filename embeds grid parameters so that runs with different grids
    for the same model do not overwrite each other.
    """
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the grid-info suffix once (same for all visits)
    # Use the vsys/kp axes from the first result's meshgrids
    vsys_ax = results[0]['vsys'][:, 0]
    kp_ax   = results[0]['kp'][0, :]
    grid_suffix = _build_grid_stem_suffix(vsys_ax, kp_ax, apply_alpha)

    # Contact phases (T1-T4) are planet-wide, computed once
    contact_phases = _compute_contact_phases(planet, kind_trans)
    log.info(
        f'Contact phases  [T1, T2, T3, T4]: '
        + ', '.join(f'{p:.4f}' for p in contact_phases)
    )

    for i, (data_tr, di, visit_res) in enumerate(zip(data_trs, data_info_list, results), start=1):
        noise = data_tr['noise'][:, idx_orders]
        flux = data_tr['flux'][:, idx_orders]
        uncert_sum = np.sum(np.ma.log(noise), axis=axis_sum)
        s2f = np.sum(flux ** 2, axis=axis_sum)

        # Phase is not stored in data_tr by load_sequences — compute it from
        # t_start and the planet orbital parameters (set by setup_logl_grid).
        t_start = data_tr['t_start']
        phase = ((t_start * u.d - planet.mid_tr) / planet.period).decompose().value
        phase -= np.round(phase.mean())
        if kind_trans == 'emission':
            # Emission: secondary eclipse at phase 0.5 → shift to [0, 1] range.
            # This matches the convention used in gen_transit_model for emission.
            if (phase < 0).all():
                phase += 1.0

        saved = dict(
            **visit_res,
            alpha_frac=di['trall_alpha_frac'],
            icorr=di['trall_icorr'],
            N=di['trall_N'],
            bad_indexs=np.empty(0),
            s2f=s2f,
            uncert_sum=uncert_sum,
            phase=phase,
            t_start=t_start,
            contact_phases=contact_phases,
            kind_trans=np.array([kind_trans]),
            apply_alpha=np.array([apply_alpha]),
        )

        for key, val in list(saved.items()):
            if not isinstance(val, np.ndarray):
                log.warning(f'{key}: not an ndarray, saving as empty.')
                saved[key] = np.empty(0)
            elif val.dtype == object:
                log.warning(f'{key}: dtype=object, saving as empty.')
                saved[key] = np.empty(0)

        filename = f'{file_stem}{grid_suffix}_visit{i}.npz'
        np.savez(output_path / filename, **saved)
        log.info(f'Saved: {output_path / filename}')


# ---------------------------------------------------------------------------
# Post-processing — load results + compute logL / CCF
# ---------------------------------------------------------------------------

# Module globals populated by load_logl_results
cross_terms = None
squared_terms = None
s2f = None
uncert_sum = None
N = None
vsys_axis = None
kp_axis = None
_loaded_extra = {}

_CONCAT_AXIS = {
    'cross_terms': -2,
    'squared_terms': -2,
    'alpha_frac': 0,
    'N': 0,
    's2f': 0,
    'uncert_sum': 0,
    't_start': 0,
    'phase': 0,
    'kp': None,
    'vsys': None,
    'icorr': 'add',
    'bad_indexs': 'add',
    # Per-planet / per-run constants — take value from first file only
    'contact_phases': None,
    'kind_trans': None,
    'apply_alpha': None,
}


def load_logl_results(file_list, path=None):
    """Load and combine logl grid NPZ files into module globals.

    After this call, ``get_logl`` and ``get_ccf`` operate without arguments.

    Parameters
    ----------
    file_list : list of str or Path
        NPZ files to load and concatenate (one per visit).
    path : str or Path, optional
        Directory containing the files.
    """
    import re
    import warnings

    global cross_terms, squared_terms, s2f, uncert_sum, N
    global vsys_axis, kp_axis, _loaded_extra

    path = Path(path) if path else Path('.')

    # Check for duplicate visit numbers before loading.
    # Each file should correspond to a distinct visit; loading two files for
    # the same visit silently doubles its exposures and corrupts the analysis.
    visit_numbers = {}
    for f in file_list:
        m = re.search(r'_visit(\d+)', Path(f).name)
        visit_num = m.group(1) if m else Path(f).stem
        if visit_num in visit_numbers:
            warnings.warn(
                f"Duplicate visit '{visit_num}' detected:\n"
                f"  already loaded: {visit_numbers[visit_num]}\n"
                f"  duplicate:      {Path(f).name}\n"
                "Each visit should appear only once. Check your glob pattern.",
                UserWarning, stacklevel=2,
            )
        else:
            visit_numbers[visit_num] = Path(f).name
    combined = {}

    for filename in file_list:
        zf = np.load(path / filename)
        for key in zf.keys():
            exp_axis = _CONCAT_AXIS.get(key, 0)
            if key in combined and exp_axis is not None:
                if exp_axis == 'add':
                    offset = combined['N'].shape[0]
                    combined[key] = np.concatenate(
                        [combined[key], zf[key] + offset], axis=0
                    )
                else:
                    combined[key] = np.concatenate(
                        [combined[key], zf[key]], axis=exp_axis
                    )
            else:
                combined[key] = zf[key]

    cross_terms = combined['cross_terms']
    squared_terms = combined['squared_terms']
    s2f = combined['s2f']
    uncert_sum = combined['uncert_sum']
    N = combined['N']
    vsys_axis = combined['vsys'][:, 0]
    kp_axis = combined['kp'][0, :]
    _loaded_extra = combined


def get_logl(alpha=1., beta=1., kind='BL', idx_orders=None, idx_exposure=None, sum_axis=None):
    """Compute logL from the loaded grid terms.

    Parameters
    ----------
    alpha : float or 1-D array-like
        Model scaling factor.  When an array is passed together with
        ``sum_axis``, a vectorised fast path pre-sums the chi² terms and
        then broadcasts alpha, avoiding a Python loop.  The output gains a
        leading alpha axis: shape ``(n_alpha, *spatial_dims)``.
        Scalar alpha preserves the original output shape (backward compatible).
    beta : float
        Noise scaling factor (Gibson prescription only).
    kind : {'BL', 'G'}
        logL prescription: 'BL' = Brogi & Line, 'G' = Gibson.
    idx_orders : array-like, optional
        Orders to include (default: all).
    idx_exposure : array-like, optional
        Exposure indices to include (default: all).
    sum_axis : int or tuple, optional
        Axis/axes over which to sum before returning.

    Returns
    -------
    np.ma.array
        Shape ``(n_alpha, *spatial_dims)`` when alpha is a 1-D array,
        ``(*spatial_dims,)`` when alpha is a scalar.
    """
    N_ma = np.ma.array(N, mask=(N == 0))

    exp_slice = slice(None) if idx_exposure is None else np.array(idx_exposure)[:, None]
    ord_slice = np.arange(N.shape[-1]) if idx_orders is None else np.array(idx_orders)
    idx = (..., exp_slice, ord_slice)

    ct = cross_terms[idx]
    st = squared_terms[idx]
    sf = s2f[idx]
    N_i = N_ma[idx]
    us = uncert_sum[idx]

    alpha_arr = np.asarray(alpha, dtype=float)
    scalar_alpha = alpha_arr.ndim == 0

    if not scalar_alpha and sum_axis is not None:
        # --- Vectorised fast path ---
        # chi²(α) = sf_sum − 2α·ct_sum + α²·st_sum is linear in the pre-summed
        # terms, so summing first is algebraically exact and avoids an intermediate
        # (n_alpha, …, n_exp, n_ord) tensor.
        ct_sum = np.ma.sum(ct,  axis=sum_axis)
        st_sum = np.ma.sum(st,  axis=sum_axis)
        sf_sum = np.ma.sum(sf,  axis=sum_axis)
        n_sum  = np.ma.sum(N_i, axis=sum_axis)
        alpha_bcast = alpha_arr.reshape((-1,) + (1,) * ct_sum.ndim)
        chi2        = sf_sum - 2*alpha_bcast*ct_sum + alpha_bcast**2*st_sum
        if kind == 'BL':
            return -n_sum / 2 * np.ma.log(chi2 / n_sum)
        elif kind == 'G':
            us_sum = np.ma.sum(us, axis=sum_axis)
            cst = -n_sum/2 * np.ma.log(2.*np.pi) - n_sum*np.log(float(beta)) - us_sum
            return cst - 0.5 * chi2 / float(beta)**2
        else:
            raise ValueError(f"logl kind must be 'BL' or 'G', got {kind!r}")

    # --- Scalar / original path (backward compatible) ---
    chi2_val = sf - 2 * alpha_arr * ct + alpha_arr**2 * st

    if sum_axis is not None:
        chi2_val = np.ma.sum(chi2_val, axis=sum_axis)
        N_i      = np.ma.sum(N_i,      axis=sum_axis)
        if kind == 'G':
            us = np.sum(us, axis=sum_axis)

    if kind == 'BL':
        return -N_i / 2 * np.ma.log(chi2_val / N_i)
    elif kind == 'G':
        cst = -N_i / 2 * np.ma.log(2. * np.pi) - N_i * np.log(beta) - us
        return cst - 0.5 * chi2_val / beta**2
    else:
        raise ValueError(f"logl kind must be 'BL' or 'G', got {kind!r}")


def get_ccf(kind='BL', idx_orders=None, idx_exposure=None, sum_axis=None):
    """Compute CCF from the loaded grid terms.

    Parameters
    ----------
    kind : {'BL', 'G'}
        Normalisation convention.
    idx_orders, idx_exposure, sum_axis : same as get_logl.

    Returns
    -------
    np.ma.array
    """
    N_ma = np.ma.array(N, mask=(N == 0))

    exp_slice = slice(None) if idx_exposure is None else np.array(idx_exposure)[:, None]
    ord_slice = np.arange(N.shape[-1]) if idx_orders is None else np.array(idx_orders)
    idx = (..., exp_slice, ord_slice)

    if kind == 'BL':
        return np.ma.sum(cross_terms[idx] / N_ma[idx], axis=sum_axis)
    elif kind == 'G':
        return np.ma.sum(cross_terms[idx], axis=sum_axis)
    else:
        raise ValueError(f"CCF kind must be 'BL' or 'G', got {kind!r}")


# ---------------------------------------------------------------------------
# Analysis helpers — posterior computation for Kp-Vsys maps
# ---------------------------------------------------------------------------

def _log_simps(log_arr, x_or_dx, use_x=False, axis=0):
    """Numerically stable integration of exp(log_arr) via Simpson's rule.

    Computing ∫ exp(f) dx directly overflows or underflows for large |f|.
    The standard log-sum-exp trick shifts by the *maximum* before exponentiating::

        log(∫ exp(f) dx) = log(∫ exp(f − c) dx) + c    where c = max(f)

    After the shift, the dominant contribution is exp(0) = 1 (no overflow) and
    all other terms are ≤ 1; any that underflow to 0 are negligible anyway.

    Parameters
    ----------
    log_arr : ndarray
        Log of the integrand.
    x_or_dx : float or 1D array
        Uniform step size (float) or coordinate array (1D array).
    use_x : bool
        If True, x_or_dx is a coordinate array; if False it is a step size.
    axis : int
        Axis along which to integrate.

    Returns
    -------
    ndarray or float — log of the integral
    """
    from scipy import integrate
    # Shift by the MAXIMUM along the integration axis (standard log-sum-exp trick).
    # Shifting by the max ensures the dominant contribution is exp(0) = 1 and all
    # other terms are ≤ 1, so there is no overflow regardless of the dynamic range.
    # Shifting by the min would require exp(max − min) for the dominant term, which
    # overflows if the logL range exceeds ~710.  Underflow of the small-contribution
    # terms (exp(min − max) → 0) is acceptable since they are negligible.
    # Normalise the input to a plain float array before doing any arithmetic.
    # np.nanmax on a masked array returns the fill_value (typically 1e20) rather
    # than the max of the valid data — which would make exp_shift = 1e20 and
    # cause all exp(logL − 1e20) terms to underflow to zero.
    # Converting masked positions to -inf avoids the fill-value contamination:
    # -inf contributes zero to the integral (exp(-inf) = 0), which is correct.
    if isinstance(log_arr, np.ma.MaskedArray):
        log_arr = log_arr.filled(-np.inf)

    exp_shift = np.nanmax(log_arr)   # ignores NaN; -inf is just the smallest value
    if not np.isfinite(exp_shift):
        # Every position is -inf or NaN (no valid data) — integral is zero.
        result_shape = list(log_arr.shape)
        del result_shape[axis]
        return np.full(result_shape, -np.inf)

    # -inf → exp(-inf) = 0: positions with no valid data contribute nothing.
    # NaN (from cubic-spline edge artefacts) also map to 0.
    arr = np.where(np.isfinite(log_arr), np.exp(log_arr - exp_shift), 0.)

    if use_x:
        result = integrate.simps(arr, x=x_or_dx, axis=axis)
    else:
        result = integrate.simps(arr, dx=x_or_dx, axis=axis)
    with np.errstate(divide='ignore'):
        return np.log(result) + exp_shift


def compute_kpvsys_posterior(alpha_array=None, idx_signal=None,
                              idx_orders=None, oversample=2, kind='BL'):
    """Compute the alpha-marginalized Kp-Vsys posterior.

    Since the model amplitude alpha is not known a priori, we marginalize over
    it rather than fixing it to 1::

        P(Kp, v_sys | data) = ∫ L(Kp, v_sys, α) dα

    Integration is done via Simpson's rule in log-space (_log_simps) for
    numerical stability.  The posterior is then oversampled by cubic spline
    interpolation before computing sigma contours: the contour integrals
    converge faster on a finer grid, so oversampling improves the accuracy
    of the sigma levels without recomputing the expensive logl grid.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to marginalize over.
        Default: 31 points linearly spaced in [0.01, 2.0].
        The range covers sub- to super-solar model amplitudes.
    idx_signal : 1D int array, optional
        Indices of exposures where the planet signal is detectable
        (alpha_frac > 0.5).  Default: derived from loaded alpha_frac.
    idx_orders : array-like, optional
        Spectral orders to include.  Default: all.
    oversample : int
        Oversampling factor applied to log(posterior) via cubic interpolation
        before computing marginals and contours.  Default: 2.
    kind : {'BL', 'G'}
        LogL prescription: 'BL' = Brogi & Line (default), 'G' = Gibson.

    Returns
    -------
    posterior : (n_vsys_os, n_kp_os) array
        Alpha-marginalized, oversampled posterior (linear, not log).
    vsys_os : (n_vsys_os,) array  — oversampled v_sys axis
    kp_os   : (n_kp_os,)  array  — oversampled Kp axis
    margin_vsys : (n_vsys_os,) array  — posterior marginalized over Kp
    margin_kp   : (n_kp_os,)   array  — posterior marginalized over v_sys
    """

    if alpha_array is None:
        # 31 points in [0.01, 2.0] gives adequate integration accuracy.
        alpha_array = np.linspace(0.01, 2., 31)

    if idx_signal is None:
        # alpha_frac is the fraction of the total planet signal received during
        # each exposure.  It equals 1 during full eclipse/transit and 0 when
        # the planet is hidden by the star.  Exposures with alpha_frac > 0.5
        # are those where we expect a detectable signal.
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    # --- Marginalise over alpha in log-space ---
    log.info(f'Marginalising over {len(alpha_array)} alpha values ...')
    logl_map_alpha = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_alpha = alpha_array[1] - alpha_array[0]
    log_posterior = _log_simps(logl_map_alpha, d_alpha, axis=0)   # (n_vsys, n_kp)

    # Normalise to max=0 before exp().
    finite_lp = log_posterior[np.isfinite(log_posterior)]
    lp_max = float(finite_lp.max()) if len(finite_lp) else 0.
    log_post_norm = log_posterior - lp_max

    # --- Optional oversampling of the 2D posterior (display only) ---
    if oversample > 1:
        from starships.plotting_fcts import oversample_image
        # Replace any residual -inf before spline interpolation.
        floor_val = log_post_norm[np.isfinite(log_post_norm)].min()
        log_post_safe = np.where(np.isfinite(log_post_norm), log_post_norm, floor_val)
        log_post_os, kp_os, vsys_os = oversample_image(
            log_post_safe, oversample, x_coords=kp_axis, y_coords=vsys_axis,
        )
        posterior = np.exp(log_post_os)
        vsys_out, kp_out = vsys_os, kp_os
        lp_for_margins = log_post_os   # per-slice max on oversampled grid
    else:
        posterior = np.exp(log_post_norm)
        vsys_out, kp_out = vsys_axis.copy(), kp_axis.copy()
        lp_for_margins = log_post_norm

    # --- Marginals with per-slice max shift ---
    # Uses a per-row/per-column max instead of the global max so every slice
    # retains its own scale and the profiles are smooth even for narrow peaks.
    margin_vsys, margin_kp = _marginals_from_log(lp_for_margins, vsys_out, kp_out)

    return posterior, vsys_out, kp_out, margin_vsys, margin_kp


def _marginals_from_log(log_post, x_axis, y_axis):
    """Compute 1D marginals from a 2D log-posterior using per-slice max shifts.

    Per-slice max (instead of the global max used by _log_simps) ensures that
    rows/columns far from the peak are not all shifted to underflow — they each
    keep their own scale and contribute a smoothly varying integral value.

    Parameters
    ----------
    log_post : (n_x, n_y) array  — log-posterior, may contain -inf
    x_axis   : (n_x,) array
    y_axis   : (n_y,) array

    Returns
    -------
    margin_x : (n_x,) array  — posterior marginalised over y  (not normalised)
    margin_y : (n_y,) array  — posterior marginalised over x  (not normalised)
    """
    from scipy import integrate as _int

    lp = np.where(np.isfinite(log_post), log_post, -np.inf)

    # margin_x(i) = ∫ exp(log_post[i, :]) dy  — integrate over y (axis=1)
    mx = np.nanmax(lp, axis=1, keepdims=True)          # (n_x, 1) — per-row max
    mx_safe = np.where(np.isfinite(mx), mx, 0.)
    shifted_x = np.where(np.isfinite(lp), np.exp(lp - mx_safe), 0.)
    intg_x = _int.simps(shifted_x, x=y_axis, axis=1)  # (n_x,)
    with np.errstate(divide='ignore'):
        margin_x = np.exp(np.log(intg_x) + mx_safe.squeeze(1))

    # margin_y(j) = ∫ exp(log_post[:, j]) dx  — integrate over x (axis=0)
    my = np.nanmax(lp, axis=0, keepdims=True)          # (1, n_y) — per-col max
    my_safe = np.where(np.isfinite(my), my, 0.)
    shifted_y = np.where(np.isfinite(lp), np.exp(lp - my_safe), 0.)
    intg_y = _int.simps(shifted_y, x=x_axis, axis=0)  # (n_y,)
    with np.errstate(divide='ignore'):
        margin_y = np.exp(np.log(intg_y) + my_safe.squeeze(0))

    return margin_x, margin_y


def _build_logl_cube(alpha_array, idx_signal, idx_orders, kind='BL'):
    """Build and pre-normalise the (n_alpha, n_vsys, n_kp) logL cube.

    Shared by compute_kpvsys_posterior, compute_alpha_kp_posterior, and
    compute_alpha_vsys_posterior so that the normalisation logic lives in one place.
    Uses the vectorised fast path in get_logl (array alpha + sum_axis).
    """
    logl_cube = get_logl(
        alpha=np.asarray(alpha_array),
        idx_orders=idx_orders,
        idx_exposure=idx_signal,
        kind=kind,
        sum_axis=(-2, -1),
    )  # shape (n_alpha, n_vsys, n_kp)
    # Pre-normalise to max=0 so that exp() calls stay within float64 range.
    if isinstance(logl_cube, np.ma.MaskedArray):
        _shift = float(logl_cube.max())
    else:
        _finite = logl_cube[np.isfinite(logl_cube)]
        _shift = float(_finite.max()) if len(_finite) > 0 else 0.
    return logl_cube - _shift


def compute_alpha_kp_posterior(alpha_array=None, idx_signal=None,
                                idx_orders=None, oversample=2, kind='BL'):
    """Compute the vsys-marginalised alpha × Kp posterior.

    P(α, Kp | data) = ∫ L(α, vsys, Kp) d(vsys)

    Useful for checking whether the model amplitude α is constrained as a
    function of Kp, independently of the system velocity.

    Parameters
    ----------
    alpha_array : 1D array, optional — alpha values (default 31 pts in [0.01, 2]).
    idx_signal : 1D int array, optional
    idx_orders : array-like, optional
    oversample : int — oversampling factor for the 2D posterior.
    kind : {'BL', 'G'} — logL prescription (default 'BL').

    Returns
    -------
    posterior : (n_alpha_os, n_kp_os) array
    alpha_os  : (n_alpha_os,) array
    kp_os     : (n_kp_os,)   array
    margin_alpha : (n_alpha_os,) array — posterior marginalised over Kp
    margin_kp    : (n_kp_os,)   array — posterior marginalised over alpha
    """

    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    logl_cube = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_vsys = vsys_axis[1] - vsys_axis[0]
    log_posterior = _log_simps(logl_cube, d_vsys, axis=1)   # (n_alpha, n_kp)

    finite_lp = log_posterior[np.isfinite(log_posterior)]
    lp_max = float(finite_lp.max()) if len(finite_lp) else 0.
    log_post_norm = log_posterior - lp_max

    if oversample > 1:
        from starships.plotting_fcts import oversample_image
        floor_val = log_post_norm[np.isfinite(log_post_norm)].min()
        log_post_safe = np.where(np.isfinite(log_post_norm), log_post_norm, floor_val)
        log_post_os, kp_os, alpha_os = oversample_image(
            log_post_safe, oversample, x_coords=kp_axis, y_coords=alpha_array,
        )
        posterior = np.exp(log_post_os)
        alpha_out, kp_out = alpha_os, kp_os
        lp_for_margins = log_post_os
    else:
        posterior = np.exp(log_post_norm)
        alpha_out, kp_out = alpha_array.copy(), kp_axis.copy()
        lp_for_margins = log_post_norm

    margin_alpha, margin_kp = _marginals_from_log(lp_for_margins, alpha_out, kp_out)

    return posterior, alpha_out, kp_out, margin_alpha, margin_kp


def compute_alpha_vsys_posterior(alpha_array=None, idx_signal=None,
                                  idx_orders=None, oversample=2, kind='BL'):
    """Compute the Kp-marginalised alpha × vsys posterior.

    P(α, vsys | data) = ∫ L(α, vsys, Kp) d(Kp)

    Parameters
    ----------
    alpha_array : 1D array, optional — alpha values (default 31 pts in [0.01, 2]).
    idx_signal : 1D int array, optional
    idx_orders : array-like, optional
    oversample : int — oversampling factor.

    Returns
    -------
    posterior : (n_alpha_os, n_vsys_os) array
    alpha_os  : (n_alpha_os,) array
    vsys_os   : (n_vsys_os,) array
    margin_alpha : (n_alpha_os,) array — posterior marginalised over vsys
    margin_vsys  : (n_vsys_os,) array — posterior marginalised over alpha
    """

    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)
    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    logl_cube = _build_logl_cube(alpha_array, idx_signal, idx_orders, kind=kind)
    d_kp = kp_axis[1] - kp_axis[0]
    log_posterior = _log_simps(logl_cube, d_kp, axis=2)   # (n_alpha, n_vsys)

    finite_lp = log_posterior[np.isfinite(log_posterior)]
    lp_max = float(finite_lp.max()) if len(finite_lp) else 0.
    log_post_norm = log_posterior - lp_max

    if oversample > 1:
        from starships.plotting_fcts import oversample_image
        floor_val = log_post_norm[np.isfinite(log_post_norm)].min()
        log_post_safe = np.where(np.isfinite(log_post_norm), log_post_norm, floor_val)
        log_post_os, vsys_os, alpha_os = oversample_image(
            log_post_safe, oversample, x_coords=vsys_axis, y_coords=alpha_array,
        )
        posterior = np.exp(log_post_os)
        alpha_out, vsys_out = alpha_os, vsys_os
        lp_for_margins = log_post_os
    else:
        posterior = np.exp(log_post_norm)
        alpha_out, vsys_out = alpha_array.copy(), vsys_axis.copy()
        lp_for_margins = log_post_norm

    margin_alpha, margin_vsys = _marginals_from_log(lp_for_margins, alpha_out, vsys_out)

    return posterior, alpha_out, vsys_out, margin_alpha, margin_vsys


def get_log_norm_posterior(param_1, param_2, post_grid):
    """Return the log-normalised posterior on a (param_1, param_2) grid.

    Subtracts the log-evidence (2D Simpson integral over the grid) from the
    log-posterior, giving a proper normalised log-probability::

        log P_norm = log P - log ∫∫ exp(log P) dp1 dp2

    Parameters
    ----------
    param_1, param_2 : 1D arrays  (e.g. vsys_axis, kp_axis)
    post_grid : (n1, n2) array  — log-posterior values (any offset)

    Returns
    -------
    (n1, n2) array — log-posterior normalised so that exp() integrates to 1
    """
    from scipy import integrate

    d1 = param_1[-1] - param_1[0]
    d2 = param_2[-1] - param_2[0]

    exp_shift = np.floor(post_grid.min())
    pg = np.exp(post_grid - exp_shift)
    log_evidence = np.log(integrate.simps(integrate.simps(pg, dx=d1), dx=d2)) + exp_shift

    return post_grid - log_evidence


def compute_alpha_marginal(alpha_array=None, idx_signal=None, idx_orders=None):
    """Compute the marginal logL as a function of alpha.

    Two quantities are returned:

    * ``logl_at_peak`` — the 1D slice of the logl cube at the peak (vsys, Kp).
      Shows how the logL varies with model amplitude at the best-fit position.
    * ``log_p_alpha`` — the full marginal posterior::

          log P(α | data) = log ∫∫ L(α, vsys, Kp) d(vsys) d(Kp)

      A detection produces a peaked marginal (alpha constrained near 1).
      A non-detection gives a flat, uninformative marginal.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to scan.  Default: 31 points in [0.01, 2.0].
    idx_signal : 1D int array, optional
        Exposures to include.  Default: alpha_frac > 0.5 from loaded data.
    idx_orders : array-like, optional
        Orders to include.  Default: all.

    Returns
    -------
    alpha_array : 1D array
    logl_at_peak : 1D array — logL at (vsys_peak, Kp_peak) for each alpha
    log_p_alpha : 1D array — log marginal posterior over alpha
    vsys_peak, kp_peak : floats — position of the alpha-marginalised peak
    """
    if alpha_array is None:
        alpha_array = np.linspace(0.01, 2., 31)

    if idx_signal is None:
        alpha_frac = _loaded_extra['alpha_frac']
        (idx_signal,) = np.nonzero(alpha_frac > 0.5)

    # Build the 3D logl cube: shape (n_alpha, n_vsys, n_kp)
    logl_map_alpha = _build_logl_cube(alpha_array, idx_signal, idx_orders)
    d_alpha = alpha_array[1] - alpha_array[0]

    # Peak position: use the alpha-marginalised map
    log_post = _log_simps(logl_map_alpha, d_alpha, axis=0)
    max_ind = np.unravel_index(np.argmax(log_post), log_post.shape)
    vsys_peak = vsys_axis[max_ind[0]]
    kp_peak   = kp_axis[max_ind[1]]

    # 1D slice at the peak
    logl_at_peak = logl_map_alpha[:, max_ind[0], max_ind[1]]

    # Full marginal P(α|data): integrate L over (vsys, Kp)
    # Step 1: integrate over Kp axis (axis=-1)
    d_kp   = kp_axis[1]   - kp_axis[0]
    d_vsys = vsys_axis[1] - vsys_axis[0]
    log_marg_kp  = _log_simps(logl_map_alpha, d_kp,   axis=2)  # (n_alpha, n_vsys)
    log_p_alpha  = _log_simps(log_marg_kp,    d_vsys, axis=1)  # (n_alpha,)

    return alpha_array, logl_at_peak, log_p_alpha, vsys_peak, kp_peak


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _find_npz_files(out_path, stem):
    """Return sorted list of NPZ files matching ``{stem}_visit*.npz``."""
    files = sorted(out_path.glob(f'{stem}_visit*.npz'))
    if not files:
        raise FileNotFoundError(
            f'No NPZ files found matching {out_path / stem}_visit*.npz\n'
            'Run without --no-compute to generate them first.'
        )
    return files


def _make_trailing_plot(out_path, stem, kp_val, rv_expected=0.,
                         noise_rv_width=15., peak_rv_width=2.):
    """Generate and save a trailing plot from existing NPZ files.

    Parameters
    ----------
    kp_val : float
        Reference Kp (km/s) at which to slice the 3D map.
    rv_expected : float
        Expected v_sys (km/s) — used to centre the noise and peak windows.
    noise_rv_width : float
        Half-width of the RV baseline window (km/s). Points *outside*
        (rv_expected ± noise_rv_width) define the noise for normalisation.
    peak_rv_width : float
        Half-width of the peak window (km/s) for the lightcurve side panel.
    """
    from starships.plotting_fcts import plot_trailing_map

    files = _find_npz_files(out_path, stem)

    # The noise window is centred on rv_expected so the normalisation baseline
    # is symmetric around the expected signal position.
    noise_rv_limits = (rv_expected - noise_rv_width, rv_expected + noise_rv_width)
    peak_rv_limits  = (rv_expected - peak_rv_width,  rv_expected + peak_rv_width)

    # is_out_rv: boolean mask selecting RV values outside the noise window.
    # These points are free of planet signal and define the per-exposure baseline.
    is_out_rv = (vsys_axis < noise_rv_limits[0]) | (noise_rv_limits[1] < vsys_axis)

    # --- Combined 1D profile (all visits, out-of-eclipse exposures only) ---
    # Load all visits first to compute the reference combined profile shown in
    # grey behind the per-visit profile.  Only out-of-eclipse exposures are
    # summed: in-eclipse exposures do not contain planet signal and would
    # dilute the detection.
    load_logl_results(files)
    alpha_frac_all = _loaded_extra['alpha_frac']
    (idx_signal_all,) = np.nonzero(alpha_frac_all > 0.5)

    # Use the nearest Kp in the grid (argmin of absolute difference), not searchsorted.
    # searchsorted returns the insertion point, which can point to the wrong neighbour
    # when kp_val falls between two grid points.
    i_kp = int(np.argmin(np.abs(kp_axis - kp_val)))
    log.info(f'Trailing plot at Kp = {kp_axis[i_kp]:.2f} km/s (requested {kp_val:.2f})')

    logl_1d_all = get_logl(idx_exposure=idx_signal_all,
                            alpha=1., sum_axis=(-2, -1))[:, i_kp]
    logl_1d_all_norm = logl_1d_all - np.ma.median(logl_1d_all[is_out_rv])
    logl_1d_all_norm /= np.ma.std(logl_1d_all_norm[is_out_rv])

    # --- Per-visit trailing maps ---
    for i_file, filename in enumerate(files, start=1):
        load_logl_results([filename])

        # Phase may be absent from the NPZ (saved as an empty array when the data
        # pipeline did not produce it).  Fall back to integer exposure index.
        phase = _loaded_extra.get('phase', None)
        if phase is None or len(phase) == 0:
            phase = np.arange(N.shape[-2])

        # Sum over spectral pixels only (axis=-1) to keep the time axis.
        # Result shape: (n_vsys, n_kp, n_exp) → sliced to (n_vsys, n_exp).
        logl_map_ts = get_logl(alpha=1., sum_axis=-1)[:, i_kp, :]

        # Out-of-eclipse mask for this visit only
        visit_alpha = _loaded_extra['alpha_frac']
        (visit_signal,) = np.nonzero(visit_alpha > 0.5)

        # 1D profile: sum over out-of-eclipse exposures and orders → (n_vsys,)
        logl_1d = get_logl(idx_exposure=visit_signal, alpha=1.,
                            sum_axis=(-2, -1))[:, i_kp]

        # Normalise trailing map to S/N units.
        # For each exposure, subtract the median logL over the out-of-signal
        # RV range, then divide by the standard deviation over that range.
        # This removes the per-exposure baseline drift and puts all exposures
        # on a common scale.
        logl_map_norm = (logl_map_ts
                         - np.ma.median(logl_map_ts[is_out_rv, :], axis=0)[None, :])
        logl_map_norm /= np.ma.std(logl_map_ts[is_out_rv, :], axis=0)[None, :]

        logl_1d_norm = logl_1d - np.ma.median(logl_1d[is_out_rv])
        logl_1d_norm /= np.ma.std(logl_1d_norm[is_out_rv])

        # Build contact-phase dict for the plot (T1/T4 = solid, T2/T3 = dash-dot)
        contact_phases = _loaded_extra.get('contact_phases', None)
        if contact_phases is not None and len(contact_phases) == 4:
            phase_contacts = {
                '1_4': [float(contact_phases[0]), float(contact_phases[3])],
                '2_3': [float(contact_phases[1]), float(contact_phases[2])],
            }
        else:
            phase_contacts = None

        save_path = out_path / f'trailing_{stem}_visit{i_file}.pdf'  # i_file starts at 1
        plot_trailing_map(
            logl_map_norm, vsys_axis, phase, logl_1d_norm,
            noise_rv_limits=noise_rv_limits,
            peak_rv_limits=peak_rv_limits,
            logl_1d_norm_all=logl_1d_all_norm,
            rv_expected=rv_expected,
            phase_contacts=phase_contacts,
            save_path=save_path,
        )
        log.info(f'Saved trailing plot: {save_path}')

    # Reload all visits so the module globals are in a consistent state
    load_logl_results(files)


def _make_kpvsys_plot(out_path, stem, alpha_array=None, oversample=2):
    """Generate and save a Kp-Vsys map from existing NPZ files.

    Parameters
    ----------
    alpha_array : 1D array, optional
        Alpha values to marginalize over (see compute_kpvsys_posterior).
    oversample : int
        Oversampling factor for the posterior (see compute_kpvsys_posterior).
    """
    from starships.plotting_fcts import plot_kpvsys_map

    files = _find_npz_files(out_path, stem)
    load_logl_results(files)

    # Full posterior computation: alpha marginalisation + oversampling + marginals
    posterior, vsys_os, kp_os, margin_vsys, margin_kp = compute_kpvsys_posterior(
        alpha_array=alpha_array, oversample=oversample,
    )

    save_path = out_path / f'kpvsys_{stem}.pdf'
    plot_kpvsys_map(posterior, vsys_os, kp_os, margin_vsys, margin_kp,
                    save_path=save_path)
    log.info(f'Saved Kp-Vsys map: {save_path}')


def main():
    """Entry point for ``run_starships_logl_grid`` (registered in setup.py).

    Usage::

        # Compute grid
        run_starships_logl_grid --config ~/scratch/.../cfg.yaml --specfile ~/scratch/.../model.npz

        # Compute + plot
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot trailing --kp 227
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot kpvsys

        # Plot only (NPZ already computed)
        run_starships_logl_grid --config cfg.yaml --specfile model.npz --plot trailing --kp 227 --no-compute
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='run_starships_logl_grid',
        description='Compute a Kp × v_sys logL grid for high-resolution spectroscopy.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', required=True, type=Path,
        help='Path to the retrieval input YAML (~ is expanded).',
    )
    parser.add_argument(
        '--specfile', required=True, type=Path,
        help='Path to the model spectrum NPZ (keys: wave, spec). ~ is expanded.',
    )
    parser.add_argument(
        '--plot', choices=['trailing', 'kpvsys'], default=None,
        help='Which figure to produce after computing (or with --no-compute).',
    )
    parser.add_argument(
        '--kp', type=float, default=0.,
        metavar='KP_KM_S',
        help='Reference Kp in km/s for the trailing plot Kp slice.',
    )
    parser.add_argument(
        '--rv', type=float, default=0.,
        metavar='RV_KM_S',
        help='Expected vsys in km/s — centres the noise window for the trailing plot.',
    )
    parser.add_argument(
        '--no-compute', action='store_true',
        help='Skip grid computation; load existing NPZ files and go straight to plotting.',
    )

    args = parser.parse_args()

    # Expand ~ in paths supplied on the command line
    config   = args.config.expanduser()
    specfile = args.specfile.expanduser()

    setup_logl_grid(config)

    out_path = (
        Path(logl_grid_output_path).expanduser()
        if logl_grid_output_path
        else Path('.')
    )
    stem = specfile.stem

    if not args.no_compute:
        load_logl_grid_data()
        load_model(specfile)
        results = compute_logl_grid()
        save_logl_grid(results, out_path, stem)

    if args.plot == 'trailing':
        _make_trailing_plot(out_path, stem, args.kp, rv_expected=args.rv)
    elif args.plot == 'kpvsys':
        _make_kpvsys_plot(out_path, stem)


if __name__ == '__main__':
    main()
