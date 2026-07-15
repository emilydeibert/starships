"""Fixed emitting-disk rotation kernel for TOI-1518b."""

import numpy as np

from astropy import units as u

from starships.spectrum import CitrusRotationKernel


VROT_MPS = 5_000.0
CITRUS_BOUNDARIES = [0.0, 0.5]
N_OVERSAMPLE = 500


def get_ker(theta_regions, tr_i=0):
    """Return one full-disk rotation kernel for one atmospheric region.

    The returned ndarray already includes both the uniformly emitting
    rotating-disk profile and the instrumental profile.
    """
    from starships import retrieval

    if len(theta_regions) != 1:
        raise ValueError(
            "The fixed TOI-1518b kernel currently supports exactly "
            f"one atmospheric region; received {len(theta_regions)}."
        )

    # unpack_theta() has already converted R_pl to cgs centimetres.
    radius_cm = np.asarray(
        theta_regions[0]["R_pl"],
        dtype=float,
    )

    if radius_cm.size != 1:
        raise ValueError(
            "Expected one planetary-radius value, received "
            f"shape {radius_cm.shape}."
        )

    radius_m = radius_cm.reshape(-1)[0].item() * 1e-2
    angular_frequency = VROT_MPS / radius_m

    kernel_object = CitrusRotationKernel(
        citrus_phases=CITRUS_BOUNDARIES,
        pl_rad=radius_m * u.m,
        omega=angular_frequency / u.s,
        resolution=retrieval.res_instru,
    )

    regional_kernels = kernel_object.resample(
        retrieval.prt_res["high"],
        phase=0.0,
        n_os=N_OVERSAMPLE,
        pad=7,
        norm=True,
    )

    # Both citrus regions contain the same atmosphere, so sum them
    # to obtain the full uniformly emitting planetary disk.
    full_disk_kernel = np.sum(
        np.asarray(regional_kernels, dtype=float),
        axis=0,
    )

    full_disk_kernel /= np.sum(full_disk_kernel)

    if not np.isfinite(full_disk_kernel).all():
        raise ValueError("Non-finite values found in rotation kernel.")

    return [full_disk_kernel]
