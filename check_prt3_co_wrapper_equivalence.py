#!/usr/bin/env python3

from collections import OrderedDict
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.modeling.physical_models import BlackBody as BB

from petitRADTRANS import physical_constants as nc
from petitRADTRANS.physics import (
    temperature_profile_function_guillot_global,
)

from starships.petitradtrans_utils import (
    gen_atm,
    retrieval_model_plain,
)


# ============================================================
# TOI-1518b parameters
# ============================================================

pressures = np.logspace(-6, 2, 100)  # bar

planet_mass = 1.83 * u.M_jupiter
planet_radius = 1.878 * u.R_jupiter
stellar_radius = 1.942 * u.R_sun
stellar_temperature = 7299.0 * u.K

gravity = (
    const.G * planet_mass / planet_radius**2
).cgs.value

temperatures = temperature_profile_function_guillot_global(
    pressures,
    10**(-1.5),   # kappa_IR [cm^2 g^-1]
    10**(1.1),    # gamma
    gravity,
    290.0,        # T_int [K]
    2420.0,       # Guillot irradiation parameter [K]
)

planet = SimpleNamespace(
    Teff=stellar_temperature,
)


# ============================================================
# Initialize pRT3 atmosphere
# ============================================================

atmosphere = gen_atm(
    species_list=["12C-16O__HITEMP"],
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


# ============================================================
# STARSHIPS wrapper calculation
# ============================================================

wave_wrapper, contrast_wrapper, abundances, mmw, vmr = (
    retrieval_model_plain(
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
        save_abundances=True,
    )
)

wave_wrapper = np.asarray(
    getattr(wave_wrapper, "value", wave_wrapper),
    dtype=float,
)

contrast_wrapper = np.asarray(
    getattr(contrast_wrapper, "value", contrast_wrapper),
    dtype=float,
)

mmw = np.asarray(
    getattr(mmw, "value", mmw),
    dtype=float,
)

if mmw.ndim == 0:
    mmw = np.full_like(temperatures, mmw, dtype=float)


# ============================================================
# Independent direct pRT3 calculation
# ============================================================

frequencies, flux_nu, extras = atmosphere.calculate_flux(
    temperatures=np.asarray(temperatures, dtype=float),
    mass_fractions=abundances,
    mean_molar_masses=mmw,
    reference_gravity=gravity,
    frequencies_to_wavelengths=False,
)

wave_direct = nc.c / frequencies / 1e-4

blackbody = BB(stellar_temperature)

star_flux_lambda = (
    blackbody(wave_direct * u.um)
    * np.pi
    * u.sr
    * const.c
    / (wave_direct * u.um)**2
).to(
    u.erg / u.cm**2 / u.s / u.cm
)

planet_flux_lambda = (
    flux_nu
    * (u.erg / u.cm**2 / u.s / u.Hz)
    * const.c
    / (wave_direct * u.um)**2
).to(
    u.erg / u.cm**2 / u.s / u.cm
)

contrast_direct = (
    planet_flux_lambda
    * (planet_radius.cgs.value / stellar_radius.cgs.value)**2
    / star_flux_lambda
).decompose().value

wave_direct = np.asarray(wave_direct, dtype=float)
contrast_direct = np.asarray(contrast_direct, dtype=float)


# ============================================================
# Compare
# ============================================================

wave_difference = wave_wrapper - wave_direct
contrast_difference = contrast_wrapper - contrast_direct

absolute_difference = np.abs(contrast_difference)

safe_denominator = np.maximum(
    np.abs(contrast_direct),
    np.finfo(float).tiny,
)

relative_difference = absolute_difference / safe_denominator

print()
print("=" * 72)
print("STARSHIPS WRAPPER VS DIRECT pRT3")
print("=" * 72)

print("Number of points:", wave_wrapper.size)

print(
    "Maximum wavelength difference [micron]:",
    np.nanmax(np.abs(wave_difference)),
)

print(
    "Maximum absolute contrast difference:",
    np.nanmax(absolute_difference),
)

print(
    "Maximum absolute difference [ppm]:",
    np.nanmax(absolute_difference) * 1e6,
)

print(
    "Maximum relative contrast difference:",
    np.nanmax(relative_difference),
)

print(
    "Median relative contrast difference:",
    np.nanmedian(relative_difference),
)

print(
    "Wrapper finite:",
    np.isfinite(contrast_wrapper).all(),
)

print(
    "Direct calculation finite:",
    np.isfinite(contrast_direct).all(),
)

np.testing.assert_allclose(
    wave_wrapper,
    wave_direct,
    rtol=0.0,
    atol=1e-12,
)

np.testing.assert_allclose(
    contrast_wrapper,
    contrast_direct,
    rtol=1e-10,
    atol=1e-14,
)

print()
print("PASS: STARSHIPS wrapper agrees with direct pRT3.")


# ============================================================
# Save comparison products
# ============================================================

np.savez(
    "prt3_co_wrapper_equivalence.npz",
    wave=wave_wrapper,
    contrast_wrapper=contrast_wrapper,
    contrast_direct=contrast_direct,
    residual=contrast_difference,
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 6),
    sharex=True,
    constrained_layout=True,
)

axes[0].plot(
    wave_wrapper,
    contrast_wrapper * 1e6,
    linewidth=0.7,
    label="STARSHIPS wrapper",
)

axes[0].plot(
    wave_direct,
    contrast_direct * 1e6,
    linewidth=0.5,
    linestyle="--",
    label="Direct pRT3",
)

axes[0].set_ylabel(r"$F_{\rm p}/F_\star$ [ppm]")
axes[0].legend()

axes[1].plot(
    wave_wrapper,
    contrast_difference * 1e6,
    linewidth=0.7,
)

axes[1].axhline(
    0.0,
    linewidth=0.7,
    linestyle=":",
)

axes[1].set_xlabel(r"Wavelength [$\mu$m]")
axes[1].set_ylabel("Residual [ppm]")

fig.savefig(
    "prt3_co_wrapper_equivalence.png",
    dpi=200,
)

plt.close(fig)

print("Saved: prt3_co_wrapper_equivalence.npz")
print("Saved: prt3_co_wrapper_equivalence.png")
