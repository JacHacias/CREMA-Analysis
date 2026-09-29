"""Resonance lineshape with the fixed instrumental broadenings of the offline line.

Two broadenings are known a priori and are held fixed in the fit:

* 60 Hz ripple on the ion energy, +/-4.5 V amplitude. The DMM read-back averages
  it away, so it cannot be corrected event by event. Bunches are released at
  50.0 Hz, so successive bunches sample the ripple 72 degrees apart and every
  dwell (>= 300 bunches) samples its phase uniformly: the energy offset follows
  the arcsine density of a randomly sampled sinusoid (the 5-phase comb matches
  its moments through fourth order). In the Doppler-corrected frame this is an
  arcsine of half-width a = nu0 |d ln D/dV| * 4.5 V, about 139 MHz for 32S at
  10.1 kV. A fit with the amplitude free returns 4.4(1.0) V on the 61k-count
  32S scan of 2026-05-08.
* Laser linewidth. The injection-seeded Ti:sapphire has ~15 MHz FWHM at the
  fundamental; frequency doubling widens its (Gaussian) spectrum by sqrt(2),
  so the probe carries 15*sqrt(2) = 21.2 MHz FWHM.

The profile is a Voigt -- free residual-Doppler Gaussian in quadrature with the
laser Gaussian, free Lorentzian -- averaged over the ripple phase with
Gauss-Chebyshev nodes, which integrate exactly against the arcsine density.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import wofz

import isotope_shift_analysis as two_fit

RIPPLE_AMPLITUDE_V = 4.5
LASER_FWHM_FUNDAMENTAL_MHZ = 15.0
LASER_FWHM_MHZ = LASER_FWHM_FUNDAMENTAL_MHZ * math.sqrt(2.0)
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
RIPPLE_NODES = 64


def voigt_area(x, sigma, gamma):
    """Area-normalized Voigt (1/x units); sigma = Gaussian sd, gamma = Lorentzian HWHM."""
    sigma = max(float(sigma), 1e-6)
    gamma = max(float(gamma), 0.0)
    z = (np.asarray(x, dtype=float) + 1j * gamma) / (sigma * math.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * math.sqrt(2.0 * math.pi))


@lru_cache(maxsize=None)
def chebyshev_nodes(n: int = RIPPLE_NODES) -> np.ndarray:
    """Nodes u_k with mean(g(u_k)) = integral of g against the arcsine density on [-1, 1]."""
    k = np.arange(1, n + 1)
    return np.cos((2 * k - 1) * np.pi / (2 * n))


def ripple_node_count(ripple_halfwidth, sigma_total) -> int:
    """Nodes for <1e-8 relative error (Gauss-Chebyshev converges exponentially in n sigma/a)."""
    return int(np.clip(math.ceil(3.2 * abs(ripple_halfwidth) / max(float(sigma_total), 1e-3)), 12, 128))


def instrument_widths(sigma_doppler, gamma, laser_fwhm_MHz=LASER_FWHM_MHZ, laser_lineshape="gaussian"):
    """Total (Gaussian sd, Lorentzian HWHM) after adding the laser linewidth."""
    laser_fwhm_MHz = max(float(laser_fwhm_MHz or 0.0), 0.0)
    if str(laser_lineshape).lower() == "lorentzian":
        return float(sigma_doppler), float(gamma) + 0.5 * laser_fwhm_MHz
    sigma_laser = laser_fwhm_MHz / FWHM_PER_SIGMA
    return math.hypot(float(sigma_doppler), sigma_laser), float(gamma)


def ripple_voigt(
    x,
    sigma_doppler,
    gamma,
    ripple_halfwidth,
    *,
    laser_fwhm_MHz=LASER_FWHM_MHZ,
    laser_lineshape="gaussian",
    nodes=None,
):
    """Area-normalized Voigt averaged over a randomly sampled sinusoidal energy ripple."""
    sigma, gamma_total = instrument_widths(sigma_doppler, gamma, laser_fwhm_MHz, laser_lineshape)
    x = np.asarray(x, dtype=float)
    a = abs(float(ripple_halfwidth))
    if a <= 0.0:
        return voigt_area(x, sigma, gamma_total)
    u = chebyshev_nodes(ripple_node_count(a, sigma)) if nodes is None else nodes
    return np.mean(voigt_area(x[..., None] - a * u, sigma, gamma_total), axis=-1)


def doppler_log_slope_per_V(
    mass_u,
    voltage_V,
    *,
    charge_e=1,
    geometry="collinear",
    neutralization="none",
    sodium_collision_branch="forward",
    dV=1.0,
):
    """|d ln D / dV| of the rest-frame Doppler factor D(V) (1/V)."""
    def log_factor(v):
        return math.log(float(two_fit.doppler_correct_ghz(
            1.0, mass_u, v, charge_e, geometry,
            neutralization=neutralization, sodium_collision_branch=sodium_collision_branch,
        )))
    return abs(log_factor(voltage_V + dV) - log_factor(voltage_V - dV)) / (2.0 * dV)


def ripple_halfwidth_MHz(nu0_MHz, mass_u, voltage_V, ripple_amplitude_V=RIPPLE_AMPLITUDE_V, **doppler_kwargs):
    """Arcsine half-width in the Doppler-corrected frame for a ripple amplitude (V)."""
    if not ripple_amplitude_V:
        return 0.0
    return float(nu0_MHz) * doppler_log_slope_per_V(mass_u, voltage_V, **doppler_kwargs) * abs(float(ripple_amplitude_V))
