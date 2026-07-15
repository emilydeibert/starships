#!/usr/bin/env python3

from pathlib import Path
from time import perf_counter

import numpy as np

# Compatibility shim for ExoFile under NumPy 2.x.
#
# ExoFile's installed by_pl_name() implementation incorrectly treats
# a one-element index array as an ambiguous result. Replace only that
# lookup method with explicit zero/one/multiple-match handling.
from exofile.archive import ExoFile


def _numpy2_safe_by_pl_name(self, *planet_names):
    catalogue_names = np.asarray(
        self["pl_name"]
    ).astype(str)

    positions = []

    for planet_name in planet_names:
        matches = np.flatnonzero(
            catalogue_names == str(planet_name)
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one catalogue row for "
                f"{planet_name!r}, found {len(matches)}."
            )

        positions.append(int(matches[0]))

    # Return a one-row Astropy table so columns retain their units.
    return self[positions]


ExoFile.by_pl_name = _numpy2_safe_by_pl_name


from starships import retrieval


YAML_FILE = (
    Path(__file__).resolve().parent
    / "retrievals"
    / "toi1518b_co_onepoint.yaml"
)

# params_prior order:
#   CO: log10 VMR
#   kp: km/s
#   rv: km/s
THETA = np.array([
    -3.0,
    207.32017922013557,
    0.0,
])


print()
print("=" * 78)
print("TOI-1518b CO ONE-POINT pRT3 LIKELIHOOD TEST")
print("=" * 78)

print("YAML:", YAML_FILE)

start_setup = perf_counter()

retrieval.setup_retrieval(YAML_FILE)

setup_seconds = perf_counter() - start_setup

parameter_order = list(retrieval.params_prior.keys())

print()
print("Parameter order:", parameter_order)
print("Theta:", THETA.tolist())

expected_order = ["CO", "kp", "rv"]

if parameter_order != expected_order:
    raise RuntimeError(
        "Unexpected parameter order. "
        f"Expected {expected_order}, got {parameter_order}."
    )


# ============================================================
# Load the real N2 reduction
# ============================================================

start_load = perf_counter()

data_info, data_sequences = retrieval.load_high_res_data()

load_seconds = perf_counter() - start_load

print()
print("Number of loaded sequences:", len(data_sequences))

for index, sequence in enumerate(data_sequences):
    print(
        f"Sequence {index} exposures:",
        len(sequence["t_start"]),
    )

print(
    "Total alpha-fraction entries:",
    len(data_info["trall_alpha_frac"]),
)


# ============================================================
# Inspect the unpacked fixed model and rotation kernel
# ============================================================

theta_regions = retrieval.unpack_theta(THETA)
theta_dict = theta_regions[0]

print()
print("CO VMR:", theta_dict["CO"])
print("Kp [km/s]:", theta_dict["kp"])
print("RV [km/s]:", theta_dict["rv"])

print(
    "Temperature range [K]:",
    float(np.nanmin(theta_dict["temperatures"])),
    float(np.nanmax(theta_dict["temperatures"])),
)

print(
    "Planet radius passed to model [cm]:",
    float(np.asarray(theta_dict["R_pl"]).reshape(-1)[0]),
)

kernels = retrieval.get_ker(
    theta_regions,
    tr_i=0,
)

print()
print("Number of returned kernels:", len(kernels))
print("Rotation-kernel points:", len(kernels[0]))
print("Rotation-kernel sum:", float(np.sum(kernels[0])))
print(
    "Rotation kernel finite:",
    bool(np.isfinite(kernels[0]).all()),
)

if len(kernels) != 1:
    raise RuntimeError(
        f"Expected one rotation kernel, received {len(kernels)}."
    )

if not np.isclose(np.sum(kernels[0]), 1.0):
    raise RuntimeError("Rotation kernel is not normalized.")


# ============================================================
# Perform exactly one complete likelihood evaluation
# ============================================================

print()
print("Evaluating one full likelihood point...")
print(
    "This includes pRT3 model generation, rotational/instrumental "
    "broadening, data forward processing, and likelihood summation."
)

start_likelihood = perf_counter()

log_probability = retrieval.lnprob(THETA)

likelihood_seconds = perf_counter() - start_likelihood

print()
print("=" * 78)
print("RESULT")
print("=" * 78)

print("Log probability:", log_probability)
print("Finite:", bool(np.isfinite(log_probability)))
print()
print("Setup time [s]:", setup_seconds)
print("Data-load time [s]:", load_seconds)
print("Likelihood time [s]:", likelihood_seconds)

if not np.isfinite(log_probability):
    raise RuntimeError(
        "The one-point likelihood returned a non-finite value."
    )

print()
print(
    "PASS: one real N2 CO likelihood evaluation completed "
    "with the pRT3 forward model."
)
