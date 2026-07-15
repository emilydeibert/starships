#!/usr/bin/env python3

from collections import OrderedDict
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as const
from astropy import units as u

from petitRADTRANS.physics import (
    temperature_profile_function_guillot_global,
)

from starships.petitradtrans_utils import (
    gen_atm,
    prepare_model,
    retrieval_model_plain,
)

from starships.spectrum import CitrusRotationKernel


# ============================================================
# Configuration
# ============================================================

R_INPUT = 100_000
R_SPIROU = 64_000

VROT = 5.0 * u.km / u.s

# Two boundaries divide the entire sphere into two hemispheres.
# Since both regions represent the same uniform atmosphere, their
# projected kernels are summed to obtain the full emitting-disk kernel.
CITRUS_BOUNDARIES = [0.0, 0.5]

# Representative phases spanning the TOI-1518b observations.
TEST_PHASES = [0.36, 0.40, 0.44, 0.62, 0.68]

PHASE_FOR_SPECTRUM = 0.40

# Oversampling of the native rotation kernel.
N_OS = 500


# ============================================================
# TOI-1518b parameters
# ============================================================

pressures = np.logspace(-6, 2, 100)  # bar

planet_mass = 1.83 * u.M_jupiter
planet_radius = 1.878 * u.R_jupiter
stellar_radius = 1.942 * u.R_sun
orbital_period = 1.90261131 * u.day

gravity = (
    const.G
    * planet_mass
    / planet_radius**2
).cgs.value

temperatures = temperature_profile_function_guillot_global(
    pressures,
    10**(-1.5),   # kappa_IR [cm^2 g^-1]
    10**(1.1),    # gamma
    gravity,
    290.0,        # T_int [K]
    2420.0,       # irradiation-temperature parameter [K]
)

planet = SimpleNamespace(
    Teff=7299.0 * u.K,
)


# ============================================================
# Rotation quantities
# ============================================================

omega = (
    VROT
    / planet_radius
).to(1 / u.s)

implied_rotation_period = (
    2.0 * np.pi / omega
).to(u.day)

synchronous_vrot = (
    2.0
    * np.pi
    * planet_radius
    / orbital_period
).to(u.km / u.s)

print()
print("=" * 76)
print("TOI-1518b ROTATION PARAMETERS")
print("=" * 76)

print(
    "Adopted vrot sin(i) [km/s]:",
    VROT.to_value(u.km / u.s),
)

print(
    "Angular frequency [1/s]:",
    omega.to_value(1 / u.s),
)

print(
    "Implied rotation period [day]:",
    implied_rotation_period.to_value(u.day),
)

print(
    "Synchronous equatorial velocity [km/s]:",
    synchronous_vrot.to_value(u.km / u.s),
)


# ============================================================
# Construct the STARSHIPS emitting-disk rotation object
# ============================================================

rotation_object = CitrusRotationKernel(
    citrus_phases=CITRUS_BOUNDARIES,
    pl_rad=planet_radius.to(u.m),
    omega=omega,
    resolution=R_SPIROU,
)


# ============================================================
# Test native Citrus kernel against analytic Gray profile
# ============================================================

native_kernels = {}
native_velocity_grids = {}

for phase in TEST_PHASES:
    velocity_grid, regional_kernels = rotation_object.get_ker(
        phase=phase,
        n_os=N_OS,
        pad=7,
        norm=True,
    )

    regional_kernels = np.asarray(
        regional_kernels,
        dtype=float,
    )

    full_disk_kernel = np.sum(
        regional_kernels,
        axis=0,
    )

    full_disk_kernel /= np.sum(full_disk_kernel)

    native_velocity_grids[phase] = np.asarray(
        velocity_grid,
        dtype=float,
    )

    native_kernels[phase] = full_disk_kernel


reference_phase = TEST_PHASES[0]
velocity_reference = native_velocity_grids[reference_phase]
kernel_reference = native_kernels[reference_phase]

vrot_mps = VROT.to_value(u.m / u.s)

gray_kernel = np.zeros_like(velocity_reference)

inside_disk = (
    np.abs(velocity_reference)
    < vrot_mps
)

gray_kernel[inside_disk] = np.sqrt(
    1.0
    - (
        velocity_reference[inside_disk]
        / vrot_mps
    )**2
)

gray_kernel /= np.sum(gray_kernel)


# Compare peak-normalized shapes.
citrus_shape = (
    kernel_reference
    / np.nanmax(kernel_reference)
)

gray_shape = (
    gray_kernel
    / np.nanmax(gray_kernel)
)

maximum_gray_shape_difference = np.nanmax(
    np.abs(
        citrus_shape
        - gray_shape
    )
)

l1_gray_difference = np.sum(
    np.abs(
        kernel_reference
        - gray_kernel
    )
)


# Test that summing all uniform regions removes phase dependence.
maximum_phase_shape_difference = 0.0

for phase in TEST_PHASES[1:]:
    velocity_phase = native_velocity_grids[phase]
    kernel_phase = native_kernels[phase]

    np.testing.assert_allclose(
        velocity_phase,
        velocity_reference,
        rtol=0.0,
        atol=1e-10,
    )

    phase_shape = (
        kernel_phase
        / np.nanmax(kernel_phase)
    )

    difference = np.nanmax(
        np.abs(
            phase_shape
            - citrus_shape
        )
    )

    maximum_phase_shape_difference = max(
        maximum_phase_shape_difference,
        difference,
    )


print()
print("=" * 76)
print("NATIVE ROTATION-KERNEL VALIDATION")
print("=" * 76)

print(
    "Citrus boundaries:",
    CITRUS_BOUNDARIES,
)

print(
    "Number of citrus regions:",
    len(CITRUS_BOUNDARIES),
)

print(
    "Kernel velocity range [km/s]:",
    float(np.nanmin(velocity_reference)) / 1e3,
    float(np.nanmax(velocity_reference)) / 1e3,
)

print(
    "Native Citrus kernel sum:",
    float(np.sum(kernel_reference)),
)

print(
    "Analytic Gray kernel sum:",
    float(np.sum(gray_kernel)),
)

print(
    "Maximum peak-normalized Citrus/Gray difference:",
    float(maximum_gray_shape_difference),
)

print(
    "L1 Citrus/Gray kernel difference:",
    float(l1_gray_difference),
)

print(
    "Maximum phase-dependent shape difference:",
    float(maximum_phase_shape_difference),
)

assert np.isfinite(kernel_reference).all()
assert np.isclose(np.sum(kernel_reference), 1.0)
assert np.isclose(np.sum(gray_kernel), 1.0)

# These tolerances allow for finite velocity-grid sampling while
# requiring the Citrus result to behave like a uniform rotating disk.
assert maximum_gray_shape_difference < 0.05, (
    "The summed Citrus kernel differs substantially from the "
    "analytic uniform-disk Gray profile."
)

assert maximum_phase_shape_difference < 0.05, (
    "The summed uniform-disk kernel unexpectedly depends on phase."
)

print()
print(
    "PASS: summed Citrus regions reproduce a uniform "
    "emitting-disk rotation kernel."
)


# ============================================================
# Generate the raw pRT3 CO emission spectrum
# ============================================================

atmosphere = gen_atm(
    species_list=[
        "12C-16O__HITEMP",
    ],
    pressures=pressures,
    mode="lbl",
    wl_range=[2.29, 2.35],
    lbl_opacity_sampling=10,
)

species = OrderedDict({
    "12C-16O__HITEMP": 1e-3,
})

species_to_linelist = {
    "CO": "12C-16O__HITEMP",
}

wave_raw, contrast_raw = retrieval_model_plain(
    atmosphere,
    species,
    planet,
    pressures,
    temperatures,
    gravity,
    0.01,
    None,
    planet_radius.cgs.value,
    stellar_radius.cgs.value,
    kind_trans="emission",
    dissociation=False,
    fct_star="blackbody",
    specie_2_lnlst=species_to_linelist,
)

wave_raw = np.asarray(
    getattr(wave_raw, "value", wave_raw),
    dtype=float,
)

contrast_raw = np.asarray(
    getattr(contrast_raw, "value", contrast_raw),
    dtype=float,
)


# ============================================================
# Instrument-only model
# ============================================================

wave_instrument, contrast_instrument = prepare_model(
    wave_raw,
    contrast_raw,
    Rbf=R_INPUT,
    Raf=R_SPIROU,
    rot_params=None,
    rot_ker=None,
)

wave_instrument = np.asarray(
    getattr(wave_instrument, "value", wave_instrument),
    dtype=float,
)

contrast_instrument = np.asarray(
    np.ma.filled(
        contrast_instrument,
        np.nan,
    ),
    dtype=float,
)


# ============================================================
# Rotation-plus-instrument kernel
# ============================================================

regional_combined_kernels = rotation_object.resample(
    R_INPUT,
    phase=PHASE_FOR_SPECTRUM,
    n_os=N_OS,
    pad=7,
    norm=True,
)

regional_combined_kernels = np.asarray(
    regional_combined_kernels,
    dtype=float,
)

rotation_instrument_kernel = np.sum(
    regional_combined_kernels,
    axis=0,
)

rotation_instrument_kernel /= np.sum(
    rotation_instrument_kernel
)


# The supplied ndarray already contains both:
#   1. the Citrus rotational profile;
#   2. the R=64,000 instrumental Gaussian.
#
# STARSHIPS therefore uses it directly instead of creating another
# instrumental Gaussian.
wave_rotation, contrast_rotation = prepare_model(
    wave_raw,
    contrast_raw,
    Rbf=R_INPUT,
    Raf=R_SPIROU,
    rot_params=None,
    rot_ker=rotation_instrument_kernel,
)

wave_rotation = np.asarray(
    getattr(wave_rotation, "value", wave_rotation),
    dtype=float,
)

contrast_rotation = np.asarray(
    np.ma.filled(
        contrast_rotation,
        np.nan,
    ),
    dtype=float,
)


# ============================================================
# Spectrum diagnostics
# ============================================================

np.testing.assert_allclose(
    wave_rotation,
    wave_instrument,
    rtol=0.0,
    atol=1e-12,
)

raw_peak_to_peak = (
    np.nanmax(contrast_raw)
    - np.nanmin(contrast_raw)
)

instrument_peak_to_peak = (
    np.nanmax(contrast_instrument)
    - np.nanmin(contrast_instrument)
)

rotation_peak_to_peak = (
    np.nanmax(contrast_rotation)
    - np.nanmin(contrast_rotation)
)

print()
print("=" * 76)
print("CO ROTATION + INSTRUMENT BROADENING TEST")
print("=" * 76)

print(
    "Raw wavelength points:",
    wave_raw.size,
)

print(
    "Final wavelength points:",
    wave_rotation.size,
)

print(
    "Combined kernel points:",
    rotation_instrument_kernel.size,
)

print(
    "Combined kernel sum:",
    float(np.sum(rotation_instrument_kernel)),
)

print()
print(
    "Raw contrast range [ppm]:",
    float(np.nanmin(contrast_raw)) * 1e6,
    float(np.nanmax(contrast_raw)) * 1e6,
)

print(
    "Instrument-only contrast range [ppm]:",
    float(np.nanmin(contrast_instrument)) * 1e6,
    float(np.nanmax(contrast_instrument)) * 1e6,
)

print(
    "Rotation-plus-instrument contrast range [ppm]:",
    float(np.nanmin(contrast_rotation)) * 1e6,
    float(np.nanmax(contrast_rotation)) * 1e6,
)

print()
print(
    "Raw peak-to-peak structure [ppm]:",
    float(raw_peak_to_peak) * 1e6,
)

print(
    "Instrument-only peak-to-peak structure [ppm]:",
    float(instrument_peak_to_peak) * 1e6,
)

print(
    "Rotation-plus-instrument peak-to-peak structure [ppm]:",
    float(rotation_peak_to_peak) * 1e6,
)

print(
    "Instrument/raw peak-to-peak ratio:",
    float(
        instrument_peak_to_peak
        / raw_peak_to_peak
    ),
)

print(
    "Rotation+instrument/raw peak-to-peak ratio:",
    float(
        rotation_peak_to_peak
        / raw_peak_to_peak
    ),
)

print(
    "Rotation+instrument/instrument-only ratio:",
    float(
        rotation_peak_to_peak
        / instrument_peak_to_peak
    ),
)

print()
print(
    "All rotation-broadened wavelengths finite:",
    bool(np.isfinite(wave_rotation).all()),
)

print(
    "All rotation-broadened contrasts finite:",
    bool(np.isfinite(contrast_rotation).all()),
)

print(
    "Rotation-broadened wavelengths increasing:",
    bool(np.all(np.diff(wave_rotation) > 0)),
)

print(
    "Instrument and rotation wavelength grids match:",
    bool(
        np.allclose(
            wave_rotation,
            wave_instrument,
            rtol=0.0,
            atol=1e-12,
        )
    ),
)

assert np.isfinite(wave_rotation).all()
assert np.isfinite(contrast_rotation).all()
assert np.all(np.diff(wave_rotation) > 0)
assert np.isclose(
    np.sum(rotation_instrument_kernel),
    1.0,
)

print()
print(
    "PASS: STARSHIPS produced a valid CO spectrum with "
    "emitting-disk rotation and the SPIRou instrumental profile."
)


# ============================================================
# Save numerical products
# ============================================================

np.savez(
    "prt3_co_emission_rotation.npz",
    velocity_native=velocity_reference,
    citrus_kernel_native=kernel_reference,
    gray_kernel_native=gray_kernel,
    rotation_instrument_kernel=rotation_instrument_kernel,
    wave_raw=wave_raw,
    contrast_raw=contrast_raw,
    wave_instrument=wave_instrument,
    contrast_instrument=contrast_instrument,
    wave_rotation=wave_rotation,
    contrast_rotation=contrast_rotation,
    vrot_kms=VROT.to_value(u.km / u.s),
    resolving_power_input=R_INPUT,
    resolving_power_instrument=R_SPIROU,
)


# ============================================================
# Save diagnostic plots
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(
    velocity_reference / 1e3,
    citrus_shape,
    linewidth=1.2,
    label="Summed Citrus kernel",
)

ax.plot(
    velocity_reference / 1e3,
    gray_shape,
    linewidth=1.0,
    linestyle="--",
    label=r"Analytic Gray profile ($\epsilon=0$)",
)

ax.set_xlim(-8, 8)
ax.set_xlabel(r"Velocity [km s$^{-1}$]")
ax.set_ylabel("Peak-normalized kernel")
ax.set_title(
    r"TOI-1518b emitting-disk rotation kernel: "
    r"$v_{\rm rot}\sin i=5$ km s$^{-1}$"
)
ax.legend()

fig.tight_layout()

fig.savefig(
    "prt3_co_emission_rotation_kernel.png",
    dpi=200,
)

plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 4.8))

ax.plot(
    wave_raw,
    contrast_raw * 1e6,
    linewidth=0.5,
    alpha=0.55,
    label=r"Raw pRT3 ($R\simeq100{,}000$)",
)

ax.plot(
    wave_instrument,
    contrast_instrument * 1e6,
    linewidth=0.8,
    label=r"Instrument only ($R=64{,}000$)",
)

ax.plot(
    wave_rotation,
    contrast_rotation * 1e6,
    linewidth=0.9,
    label=(
        r"Rotation + instrument "
        r"($v_{\rm rot}\sin i=5$ km s$^{-1}$)"
    ),
)

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$F_{\rm p}/F_\star$ [ppm]")
ax.set_title(
    "TOI-1518b CO emission broadening"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "prt3_co_emission_rotation_spectrum.png",
    dpi=200,
)

plt.close(fig)

print()
print(
    "Saved: prt3_co_emission_rotation.npz"
)

print(
    "Saved: prt3_co_emission_rotation_kernel.png"
)

print(
    "Saved: prt3_co_emission_rotation_spectrum.png"
)
