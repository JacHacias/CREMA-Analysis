"""Exposure-normalized, dead-time-corrected rate spectra and their Poisson fit.

This is analysis model "v2". Per isotope block (one or more consecutive scans):

1. Every row is kept (empty-bunch rows carry the exposure). MagneTOF echo counts
   are tagged per bunch before any ToF gating (counting_corrections).
2. One entry per bunch: HeNe-corrected laser frequency, DMM voltage and dwell
   (scan pass x scan_bin_index). Echo-cleaned in-gate hits are attributed to it.
3. Dead time: live fraction of in-gate ions per dwell from same-bunch hit pairs.
4. Groups = dwell x lab-frequency cell (10 MHz): bunches N, in-gate counts n,
   live fraction L, mean frequency and voltage.
5. n ~ Poisson(N L R(x)) with R(x) = b0 + b1 (x - xm) + A P(x - x0), P the
   ripple-averaged Voigt with the laser linewidth (ripple_lineshape). Maximum
   likelihood via Poisson-deviance residuals; covariance from the expected
   Fisher information.

Fitting per-bunch rates against the bunch exposure removes the laser-dwell comb
of the raw count spectrum (bunches per 10 MHz vary 3x between neighboring bins)
without the noise amplification of dividing sparse counts by shots. Frequencies
are MHz internally; results use the GHz keys of the legacy fitters.

Optional energy-loss tail (tail_model "exponential"): a fraction f of the atoms
carries an extra kinetic-energy loss with an exponential distribution of mean
lambda, so P becomes (1 - f) P + f (P convolved with the one-sided exponential),
on the low-frequency side in collinear geometry. "center" is then the no-loss
component. The profile is built by FFT on a 1 MHz grid (Gaussian x Lorentzian x
J0 ripple factor x exponential factor). shape_transfer carries the reference
isotope's tail (and optionally Gaussian width), converted to volts, into the fit
of the other isotope of a pair.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import j0

import counting_corrections as cc
import isotope_shift_analysis as two_fit
import ripple_lineshape as rl
from plot_style import apply_publication_style, style_axes

MHZ_PER_CM = two_fit.C * 100.0 * 1e-6
PARAM_NAMES = ("amplitude", "center", "sigma_doppler", "gamma", "background", "slope", "ripple_halfwidth",
               "tail_fraction", "tail_length")
TAIL_MODELS = ("none", "exponential")
TAIL_MAX_FRACTION = 0.95
TAIL_MAX_LENGTH_MHZ = 1500.0
FFT_GRID_MHZ = 1.0
FFT_PAD_MHZ = 4000.0

V2_DEFAULTS: dict[str, Any] = {
    "analysis_model": "v2",
    "remove_echo_counts": True,
    "echo_windows_ns": [list(w) for w in cc.ECHO_WINDOWS_NS],
    "deadtime_correction": True,
    "exposure_normalization": True,
    "ripple_amplitude_V": rl.RIPPLE_AMPLITUDE_V,
    "fit_ripple_amplitude": False,
    "laser_linewidth_fwhm_MHz": round(rl.LASER_FWHM_MHZ, 3),
    "laser_lineshape": "gaussian",
    "lorentzian_hwhm_MHz": None,
    "group_cell_MHz": 10.0,
    # Centroid uncertainty from a block bootstrap over scan passes (replicas 0 =
    # Fisher only; block_bins > 0 splits passes into segments of that many steps).
    "bootstrap_replicas": 200,
    "bootstrap_block_bins": 0,
    "bootstrap_seed": 20260928,
    # Energy-loss tail. None = free; a number fixes it (tail_length_V and
    # sigma_doppler_V in volts of beam energy, converted with the Doppler slope).
    "tail_model": "none",
    "tail_fraction": None,
    "tail_length_V": None,
    "sigma_doppler_V": None,
    "fit_background_slope": True,
    # "none" | "tail" (tail fraction + length) | "tail+sigma" (also the Gaussian
    # width): taken from the reference isotope's fit, fixed in the other's.
    "shape_transfer": "none",
    # Bunches whose wavemeter reading is this far from the laser target (cm-1)
    # are dropped (read-out glitches); 0 disables.
    "wavemeter_glitch_tol_cm": 0.01,
}
V2_OPTION_KEYS = tuple(V2_DEFAULTS)
NULLABLE_V2_KEYS = frozenset({"lorentzian_hwhm_MHz", "tail_fraction", "tail_length_V", "sigma_doppler_V"})


def is_v2(options: dict[str, Any]) -> bool:
    return str(options.get("analysis_model", "legacy")).lower() == "v2"


def v2_option(options: dict[str, Any], key: str):
    value = options.get(key, V2_DEFAULTS[key])
    return V2_DEFAULTS[key] if value is None and key not in NULLABLE_V2_KEYS else value


def echo_windows(options: dict[str, Any]):
    if not v2_option(options, "remove_echo_counts"):
        return ()
    return tuple((float(lo), float(hi)) for lo, hi in v2_option(options, "echo_windows_ns"))


# ---------------------------------------------------------------------------
# Loading and spectrum construction
# ---------------------------------------------------------------------------

def hit_mask(frame: pd.DataFrame) -> np.ndarray:
    return (frame["channel"].to_numpy() == 2) & (frame["tof"].to_numpy() > 0)


def load_scan_frame(paths, windows_ns=cc.ECHO_WINDOWS_NS) -> pd.DataFrame:
    """Concatenate raw scan CSVs (all rows) with file_index and an is_echo flag.

    Echoes are tagged per file because tagger bunch ids restart between sessions.
    """
    frames = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(Path(path))
        frame["file_index"] = index
        is_echo = np.zeros(len(frame), dtype=bool)
        if windows_ns:
            hits = hit_mask(frame)
            is_echo[hits] = cc.tag_echo_hits(frame["bunch_id"].to_numpy()[hits], frame["tof"].to_numpy()[hits], windows_ns)
        frame["is_echo"] = is_echo
        frames.append(frame)
    if not frames:
        raise ValueError("At least one file is required.")
    columns = list(frames[0].columns)
    for path, frame in zip(paths[1:], frames[1:]):
        if list(frame.columns) != columns:
            raise ValueError(f"File {path} has columns {list(frame.columns)}, expected {columns}.")
    return pd.concat(frames, ignore_index=True)


@dataclass
class RateSpectrum:
    """Grouped exposure/count data of one isotope block, ready to fit."""

    label: str
    gate_us: tuple[float, float]
    nu_lab_MHz: np.ndarray
    voltage_V: np.ndarray
    bunches: np.ndarray
    counts: np.ndarray
    live: np.ndarray
    hit_tof_s: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Scan structure of every group, for the block bootstrap: file, pass within the
    # file, scan_bin_index (laser step).
    group_file: np.ndarray | None = None
    group_pass: np.ndarray | None = None
    group_bin: np.ndarray | None = None

    @property
    def num_points(self) -> int:
        return int(self.counts.sum())

    def subset(self, index: np.ndarray) -> "RateSpectrum":
        """Groups selected by index (repeats allowed), for bootstrap replicas."""
        pick = lambda a: None if a is None else a[index]
        return RateSpectrum(
            label=self.label, gate_us=self.gate_us, nu_lab_MHz=self.nu_lab_MHz[index],
            voltage_V=self.voltage_V[index], bunches=self.bunches[index], counts=self.counts[index],
            live=self.live[index], hit_tof_s=self.hit_tof_s, diagnostics=self.diagnostics,
            group_file=pick(self.group_file), group_pass=pick(self.group_pass), group_bin=pick(self.group_bin),
        )

    @property
    def size(self) -> int:  # legacy code reports dat.size as the gated point count
        return self.num_points


def _lab_frequency_MHz(frame: pd.DataFrame, options: dict[str, Any]) -> np.ndarray:
    wn_col = options.get("wn_col", "wavemeter_wn1")
    wn = frame[wn_col].to_numpy(dtype=float)
    hene_col = options.get("hene_col", "wavemeter_wn4")
    if options.get("use_hene_calibration", False) and hene_col in frame.columns:
        hene = frame[hene_col].to_numpy(dtype=float)
        reference = options.get("hene_reference_wn")
        if reference is None:
            reference = two_fit.hene_reference_wavenumber_cm(
                hene_reference_wavelength_nm=options.get("hene_reference_wavelength_nm", two_fit.DEFAULT_HENE_WAVELENGTH_NM),
                hene_reference_wavelength_medium=options.get("hene_reference_wavelength_medium", "vacuum"),
                hene_wavenumber_medium=options.get("hene_wavenumber_medium", "vacuum"),
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            wn = np.where(np.isfinite(hene), wn * (float(reference) / hene), np.nan)
    return wn * float(options.get("frequency_multiplier", 2.0)) * MHZ_PER_CM


def _voltage_V(frame: pd.DataFrame, options: dict[str, Any]) -> np.ndarray:
    col = options.get("voltage_col", "voltage")
    if options.get("use_voltage_column", True) and col in frame.columns:
        return frame[col].to_numpy(dtype=float) * float(options.get("voltage_multiplier", two_fit.B_HVD2))
    return np.full(len(frame), float(options.get("beam_voltage_V", 10000.0)))


def _dwell_codes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Dwell = (file, scan pass, scan_bin_index); passes restart when the bin index drops.

    Returns the dwell code of every row and a (n_dwells, 3) array of (file, pass, bin).
    """
    bins = frame["scan_bin_index"].to_numpy(dtype=float)
    files = frame["file_index"].to_numpy()
    drop = np.r_[False, (np.diff(bins) < 0) | (np.diff(files) != 0)]
    passes = np.cumsum(drop)
    codes, uniques = pd.factorize(pd.MultiIndex.from_arrays([files, passes, bins]))
    return codes, np.array([list(u) for u in uniques], dtype=float).reshape(-1, 3)


def build_rate_spectrum(frame: pd.DataFrame, label: str, gate_us, options: dict[str, Any]) -> RateSpectrum:
    """Group a (bad-scan-filtered, echo-tagged) frame into exposure/count cells."""
    if gate_us is None:
        raise ValueError(f"Analysis v2 needs a ToF gate for {label}.")
    gate_us = (float(gate_us[0]), float(gate_us[1]))
    nu_lab = _lab_frequency_MHz(frame, options)
    voltage = _voltage_V(frame, options)
    finite = np.isfinite(nu_lab) & np.isfinite(voltage)
    glitch_tol = float(v2_option(options, "wavemeter_glitch_tol_cm") or 0.0)
    wn_col = options.get("wn_col", "wavemeter_wn1")
    glitches = np.zeros(len(frame), dtype=bool)
    if glitch_tol > 0 and "laser_target_wn" in frame.columns and wn_col in frame.columns:
        offset_cm = frame[wn_col].to_numpy(dtype=float) - frame["laser_target_wn"].to_numpy(dtype=float)
        glitches = np.abs(offset_cm) > glitch_tol
        finite &= ~glitches
    n_glitch_bunches = int(frame.loc[glitches, "bunch_id"].nunique()) if glitches.any() else 0
    frame = frame.loc[finite].reset_index(drop=True)
    nu_lab, voltage = nu_lab[finite], voltage[finite]

    dwell_of_row, dwell_keys = _dwell_codes(frame)
    bunch_key = pd.MultiIndex.from_arrays([frame["file_index"].to_numpy(), frame["bunch_id"].to_numpy()])
    bunch_of_row, _ = pd.factorize(bunch_key)
    n_bunches = int(bunch_of_row.max()) + 1 if bunch_of_row.size else 0
    first_row = np.full(n_bunches, -1, dtype=np.int64)
    first_row[bunch_of_row[::-1]] = np.arange(bunch_of_row.size)[::-1]
    bunch_nu = nu_lab[first_row]
    bunch_voltage = voltage[first_row]
    bunch_dwell = dwell_of_row[first_row]
    n_dwells = int(bunch_dwell.max()) + 1 if bunch_dwell.size else 0

    hits = hit_mask(frame)
    echo = frame["is_echo"].to_numpy(dtype=bool) if "is_echo" in frame.columns else np.zeros(len(frame), bool)
    clean = hits & ~echo
    tof_s = frame["tof"].to_numpy(dtype=float)
    tof_us = tof_s * 1e6
    in_gate = clean & (tof_us > gate_us[0]) & (tof_us < gate_us[1])

    windows = echo_windows(options)
    if v2_option(options, "deadtime_correction"):
        coverage = cc.dead_coverage_function(tof_s[in_gate], gate_us, windows_ns=windows)
        veto_coverage = cc.dead_coverage_function(tof_s[in_gate], gate_us, windows_ns=windows, veto_only=True)
        tagged = hits & echo
        live_dwell, _, _ = cc.dwell_live_fractions(
            bunch_of_row[clean], tof_s[clean], in_gate[clean], bunch_dwell, n_dwells, coverage,
            echo_bunch_code=bunch_of_row[tagged], echo_tof_s=tof_s[tagged], veto_coverage=veto_coverage,
        )
    else:
        live_dwell = np.ones(n_dwells)

    cell_MHz = float(v2_option(options, "group_cell_MHz"))
    bunch_cell = np.floor(bunch_nu / cell_MHz).astype(np.int64)
    group_of_bunch, _ = pd.factorize(pd.MultiIndex.from_arrays([bunch_dwell, bunch_cell]))
    n_groups = int(group_of_bunch.max()) + 1 if group_of_bunch.size else 0
    n_b = np.bincount(group_of_bunch, minlength=n_groups).astype(float)
    counts_b = np.bincount(bunch_of_row[in_gate], minlength=n_bunches).astype(float)
    counts = np.bincount(group_of_bunch, weights=counts_b, minlength=n_groups)
    mean_nu = np.bincount(group_of_bunch, weights=bunch_nu, minlength=n_groups) / n_b
    mean_voltage = np.bincount(group_of_bunch, weights=bunch_voltage, minlength=n_groups) / n_b
    group_dwell = np.zeros(n_groups, dtype=np.int64)
    group_dwell[group_of_bunch] = bunch_dwell
    live = live_dwell[group_dwell]

    dwell_bunches = np.bincount(bunch_dwell, minlength=n_dwells)
    dwell_counts = np.bincount(bunch_dwell, weights=counts_b, minlength=n_dwells)
    with np.errstate(divide="ignore", invalid="ignore"):
        dwell_rate = np.where(dwell_bunches > 0, dwell_counts / dwell_bunches, 0.0)
        corrected = float(np.sum(counts / live))
    raw_hits = int(hits.sum())
    diagnostics = {
        "raw_hits": raw_hits,
        "echo_hits_removed": int((hits & echo).sum()),
        "echo_fraction": float((hits & echo).sum() / raw_hits) if raw_hits else 0.0,
        "in_gate_hits": int(in_gate.sum()),
        "bunches": int(n_bunches),
        "dwells": int(n_dwells),
        "groups": int(n_groups),
        "peak_dwell_rate_per_bunch": float(dwell_rate.max(initial=0.0)),
        "min_live_fraction": float(live_dwell[dwell_counts > 0].min(initial=1.0)),
        "deadtime_count_correction": float(corrected / counts.sum() - 1.0) if counts.sum() > 0 else 0.0,
        "effective_dead_time_ns": cc.effective_dead_time_ns(windows),
        "deadtime_correction": bool(v2_option(options, "deadtime_correction")),
        "echo_windows_ns": [list(w) for w in windows],
        "wavemeter_glitch_bunches": n_glitch_bunches,
    }
    return RateSpectrum(
        label=label,
        gate_us=gate_us,
        nu_lab_MHz=mean_nu,
        voltage_V=mean_voltage,
        bunches=n_b,
        counts=counts,
        live=live,
        hit_tof_s=tof_s[in_gate],
        diagnostics=diagnostics,
        group_file=dwell_keys[group_dwell, 0].astype(np.int64),
        group_pass=dwell_keys[group_dwell, 1].astype(np.int64),
        group_bin=dwell_keys[group_dwell, 2].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Poisson fit
# ---------------------------------------------------------------------------

@dataclass
class LineshapeFit:
    params: dict[str, float]
    errors: dict[str, float]
    deviance: float
    n_data: int
    n_free: int
    success: bool
    message: str
    ripple_fixed: bool
    x_mean: float
    tail_model: str = "none"
    tail_side: int = -1
    deviance_symmetric: float | None = None
    covariance: dict[str, dict[str, float]] | None = None
    center_unc_symmetric: float | None = None


def _deviance_residuals(n, mu):
    mu = np.clip(mu, 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(n > 0, n * np.log(n / mu), 0.0)
    dev = np.clip(2.0 * (mu - n + term), 0.0, None)
    return np.sign(n - mu) * np.sqrt(dev)


def _binned(x, n, expo, width):
    edges = np.arange(x.min() - 0.5 * width, x.max() + width, width)
    idx = np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2)
    nb = np.bincount(idx, weights=n, minlength=edges.size - 1)
    eb = np.bincount(idx, weights=expo, minlength=edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, nb, eb, idx


class ProfileGrid:
    """Uniform grid on which the tail line shape is synthesized by FFT.

    In Fourier space the profile is a product: Gaussian exp(-s^2 k^2/2) (residual
    Doppler + laser), Lorentzian exp(-g|k|), ripple J0(a k) (arcsine density), and
    for the energy-loss tail (1 - f) + f / (1 + i side k lambda) (one-sided
    exponential), times the shift exp(-i k x0). The grid spans the cells plus
    FFT_PAD_MHZ on both sides, so the periodic images of the line and its tail
    stay negligible.
    """

    def __init__(self, x, h: float = None, pad: float = None):
        h = FFT_GRID_MHZ if h is None else float(h)
        pad = FFT_PAD_MHZ if pad is None else float(pad)
        lo = float(np.min(x)) - pad
        span = float(np.max(x)) + pad - lo
        n = 1 << int(math.ceil(math.log2(max(span / h, 16.0))))
        self.lo, self.h, self.n = lo, h, n
        self.grid = lo + h * np.arange(n)
        self.k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
        self._ripple = (None, None)

    def _ripple_factor(self, a: float) -> np.ndarray:
        if self._ripple[0] != a:
            self._ripple = (a, j0(a * self.k) if a > 0 else np.ones_like(self.k))
        return self._ripple[1]

    def profile(self, x, center, sigma_doppler, gamma, ripple_halfwidth, tail_fraction, tail_length, *,
                laser_fwhm_MHz, laser_lineshape="gaussian", tail_side=-1):
        sigma, gamma_total = rl.instrument_widths(sigma_doppler, gamma, laser_fwhm_MHz, laser_lineshape)
        k = self.k
        spectrum = np.exp(-0.5 * (sigma * k) ** 2 - max(gamma_total, 0.0) * np.abs(k)) * self._ripple_factor(abs(float(ripple_halfwidth)))
        if tail_fraction > 0.0:
            spectrum = spectrum * ((1.0 - tail_fraction) + tail_fraction / (1.0 + 1j * tail_side * k * max(tail_length, 1e-6)))
        p = np.real(np.fft.ifft(spectrum * np.exp(-1j * k * (center - self.lo)))) / self.h
        return np.interp(np.asarray(x, dtype=float), self.grid, p)


def line_profile(x, params: dict[str, float], *, laser_fwhm_MHz, laser_lineshape="gaussian",
                 tail_model="none", tail_side=-1, grid: ProfileGrid | None = None):
    """Area-normalized line profile at x for fit parameters (tail ignored for tail_model none)."""
    x = np.asarray(x, dtype=float)
    if tail_model == "none":
        return rl.ripple_voigt(x - params["center"], params["sigma_doppler"], params["gamma"],
                               params["ripple_halfwidth"], laser_fwhm_MHz=laser_fwhm_MHz, laser_lineshape=laser_lineshape)
    grid = ProfileGrid(x) if grid is None else grid
    return grid.profile(x, params["center"], params["sigma_doppler"], params["gamma"], params["ripple_halfwidth"],
                        params.get("tail_fraction", 0.0), params.get("tail_length", 0.0),
                        laser_fwhm_MHz=laser_fwhm_MHz, laser_lineshape=laser_lineshape, tail_side=tail_side)


def fit_poisson_lineshape(
    x,
    n,
    expo,
    *,
    ripple_halfwidth: float,
    laser_fwhm_MHz: float,
    laser_lineshape: str = "gaussian",
    fit_ripple: bool = False,
    fixed_gamma: float | None = None,
    start: dict[str, float] | None = None,
    tail_model: str = "none",
    tail_side: int = -1,
    fixed_tail_fraction: float | None = None,
    fixed_tail_length: float | None = None,
    fixed_sigma: float | None = None,
    fit_slope: bool = True,
) -> LineshapeFit:
    """ML fit of n ~ Poisson(expo * R(x)) with the ripple-averaged Voigt profile.

    ``start`` (parameters of an earlier fit) replaces the multi-start search with a
    single warm start, as used by the bootstrap refits. With tail_model
    "exponential" and a free tail, the symmetric fit runs first and seeds a
    second multi-start search over the no-loss center, tail fraction and length.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    expo = np.asarray(expo, dtype=float)
    keep = expo > 0
    x, n, expo = x[keep], n[keep], expo[keep]
    if x.size < 8 or n.sum() < 5:
        raise ValueError("Too few populated frequency cells to fit.")
    if tail_model not in TAIL_MODELS:
        raise ValueError(f"tail_model must be one of {TAIL_MODELS}, got {tail_model!r}")
    with_tail = tail_model != "none"
    x_mean = float(np.average(x, weights=expo))
    span = float(x.max() - x.min())
    grid = ProfileGrid(x) if with_tail else None

    def rate(theta, xx):
        amp, x0, sd, gam, b0, b1, a, f, lam = theta
        if with_tail:
            profile = grid.profile(xx, x0, sd, gam, a, f, lam, laser_fwhm_MHz=laser_fwhm_MHz,
                                   laser_lineshape=laser_lineshape, tail_side=tail_side)
        else:
            profile = rl.ripple_voigt(xx - x0, sd, gam, a, laser_fwhm_MHz=laser_fwhm_MHz, laser_lineshape=laser_lineshape)
        return b0 + b1 * (xx - x_mean) + amp * profile

    def expected(theta):
        return expo * np.clip(rate(theta, x), 1e-12, None)

    # Starting values from the binned rate spectrum.
    centers, nb, eb, _ = _binned(x, n, expo, 20.0)
    ok = eb > 0
    r = np.where(ok, nb / np.where(ok, eb, 1.0), 0.0)
    smooth = gaussian_filter1d(np.where(ok, r, np.interp(centers, centers[ok], r[ok])), 2.0)
    i_max = int(np.argmax(smooth))
    b0_guess = float(np.percentile(r[ok], 10))
    peak = max(float(smooth[i_max]) - b0_guess, 1e-6)
    above = np.flatnonzero(smooth > b0_guess + 0.5 * peak)
    fwhm = max(float(centers[above.max()] - centers[above.min()]) if above.size else 100.0, 40.0)
    sigma_laser = rl.instrument_widths(0.0, 0.0, laser_fwhm_MHz, laser_lineshape)[0]
    var = (fwhm / rl.FWHM_PER_SIGMA) ** 2 - 0.5 * ripple_halfwidth ** 2 - sigma_laser ** 2
    sd_guess = math.sqrt(max(var, 20.0 ** 2))
    amp_guess = peak * fwhm * 1.064

    gamma_free = fixed_gamma is None
    sigma_free = fixed_sigma is None
    tail_fixed = with_tail and fixed_tail_fraction is not None and fixed_tail_length is not None
    tail_free = with_tail and not tail_fixed
    base = np.array([amp_guess, centers[i_max], sd_guess if sigma_free else float(fixed_sigma),
                     10.0 if gamma_free else float(fixed_gamma), b0_guess, 0.0, float(ripple_halfwidth),
                     float(fixed_tail_fraction) if tail_fixed else 0.0,
                     float(fixed_tail_length) if tail_fixed else 100.0])
    lower_all = np.array([0.0, x.min(), 0.0, 0.0, -np.inf, -np.inf, 0.0, 0.0, 5.0])
    upper_all = np.array([np.inf, x.max(), span, span, np.inf, np.inf, max(3.0 * ripple_halfwidth, 400.0),
                          TAIL_MAX_FRACTION, TAIL_MAX_LENGTH_MHZ])
    steps_all = np.array([None, 0.05, 0.05, 0.05, 1e-6, 1e-9, 0.05, 1e-4, 0.05], dtype=object)

    def free_indices(include_tail: bool) -> np.ndarray:
        idx = [0, 1] + ([2] if sigma_free else []) + ([3] if gamma_free else []) + [4] + ([5] if fit_slope else [])
        idx += [6] if fit_ripple else []
        idx += [7, 8] if include_tail else []
        return np.array(idx)

    def solve(theta_start: np.ndarray, free_idx: np.ndarray, starts_free: list[np.ndarray]):
        lower, upper = lower_all[free_idx], upper_all[free_idx]
        finite_lo, finite_hi = np.isfinite(lower), np.isfinite(upper)
        inner_lo = np.where(finite_lo, lower + 1e-9 * (np.abs(np.where(finite_lo, lower, 0.0)) + 1.0), lower)
        inner_hi = np.where(finite_hi, upper - 1e-9 * (np.abs(np.where(finite_hi, upper, 0.0)) + 1.0), upper)

        def full(free):
            theta = theta_start.copy()
            theta[free_idx] = free
            return theta

        best, error = None, None
        for x_start in starts_free:
            try:
                res = least_squares(lambda f: _deviance_residuals(n, expected(full(f))),
                                    np.clip(x_start, inner_lo, inner_hi),
                                    bounds=(lower, upper), x_scale="jac", max_nfev=4000)
            except Exception as exc:  # try the remaining starts
                error = exc
                continue
            cost = float(np.sum(res.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, res)
        if best is None:
            raise RuntimeError(f"Poisson lineshape fit failed from every start: {error}")
        return best[0], best[1], full(best[1].x), lower, upper

    def fisher_errors(theta, res_x, free_idx, lower, upper):
        """Expected Fisher information over parameters not pinned at a bound."""
        mu = expected(theta)
        at_bound = (np.abs(res_x - lower) < 1e-7 * (np.abs(lower) + 1)) | (np.abs(res_x - upper) < 1e-7 * (np.abs(upper) + 1))
        active = free_idx[~at_bound]
        steps = np.maximum(1e-4 * np.abs(theta), np.array([1e-4 * max(theta[0], 1e-3), *steps_all[1:]], dtype=float))
        jac = np.empty((mu.size, active.size))
        for k, j in enumerate(active):
            hi, lo = theta.copy(), theta.copy()
            hi[j] += steps[j]
            lo[j] -= steps[j]
            jac[:, k] = (expected(hi) - expected(lo)) / (2.0 * steps[j])
        fisher = jac.T @ (jac / mu[:, None])
        try:
            cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(fisher)
        errors = {name: 0.0 for name in PARAM_NAMES}
        for k, j in enumerate(active):
            errors[PARAM_NAMES[j]] = float(math.sqrt(max(cov[k, k], 0.0)))
        covariance = {PARAM_NAMES[j]: {PARAM_NAMES[i]: float(cov[kk, k]) for kk, i in enumerate(active)}
                      for k, j in enumerate(active)}
        return errors, covariance

    free_idx = free_indices(tail_free)
    starts = []
    center_unc_sym = None
    if start is not None:
        base = np.array([float(start.get(name, base[i])) for i, name in enumerate(PARAM_NAMES)])
        base[6] = float(start["ripple_halfwidth"]) if fit_ripple else float(ripple_halfwidth)
        if not gamma_free:
            base[3] = float(fixed_gamma)
        if not sigma_free:
            base[2] = float(fixed_sigma)
        if not fit_slope:
            base[5] = 0.0
        if tail_fixed:
            base[7], base[8] = float(fixed_tail_fraction), float(fixed_tail_length)
        elif not with_tail:
            base[7] = 0.0
        starts.append(base[free_idx])
        cost, res, theta, lower, upper = solve(base, free_idx, starts)
        deviance_sym = None
    else:
        # Symmetric (or fixed-tail) multi-start, as in the original analysis.
        stage_idx = free_indices(False)
        for dx in (0.0, -0.25 * fwhm, 0.25 * fwhm):
            for gam in ((10.0, 60.0) if gamma_free else (float(fixed_gamma),)):
                trial = base.copy()
                trial[1] = centers[i_max] + dx
                trial[3] = gam
                starts.append(trial[stage_idx])
        cost, res, theta, lower, upper = solve(base, stage_idx, starts)
        deviance_sym = cost if not tail_fixed else None
        if tail_free:
            # Counting precision of the data, independent of the tail decomposition
            # (the library's fit-uncertainty cut tests this, not the core error).
            center_unc_sym = fisher_errors(theta, res.x, stage_idx, lower, upper)[0]["center"]
            # Seed the tail search from the symmetric solution: the no-loss core
            # sits on the steep (high-energy) edge, the tail carries the rest.
            core_width = rl.FWHM_PER_SIGMA * math.hypot(theta[2], sigma_laser) + 2.0 * theta[3]
            tail_starts, seed = [], theta.copy()
            for dc in (0.1, 0.3):
                for f0 in (0.3, 0.55):
                    for lam0 in (0.6, 1.2):
                        trial = seed.copy()
                        trial[1] = seed[1] - tail_side * dc * core_width
                        trial[7] = f0
                        trial[8] = min(lam0 * core_width, 0.9 * TAIL_MAX_LENGTH_MHZ)
                        trial[5] = 0.0 if fit_slope else trial[5]
                        tail_starts.append(trial[free_idx])
            cost, res, theta, lower, upper = solve(seed, free_idx, tail_starts)
        else:
            free_idx = stage_idx

    errors, covariance = fisher_errors(theta, res.x, free_idx, lower, upper)
    return LineshapeFit(
        params={name: float(v) for name, v in zip(PARAM_NAMES, theta)},
        errors=errors,
        deviance=cost,
        n_data=int(x.size),
        n_free=int(free_idx.size),
        success=bool(res.success),
        message=str(res.message),
        ripple_fixed=not fit_ripple,
        x_mean=x_mean,
        tail_model=tail_model,
        tail_side=int(tail_side),
        deviance_symmetric=deviance_sym,
        covariance=covariance,
        center_unc_symmetric=center_unc_sym,
    )


def model_rate(fit: LineshapeFit, x, *, laser_fwhm_MHz, laser_lineshape="gaussian"):
    p = fit.params
    x = np.asarray(x, dtype=float)
    profile = line_profile(x, p, laser_fwhm_MHz=laser_fwhm_MHz, laser_lineshape=laser_lineshape,
                           tail_model=fit.tail_model, tail_side=fit.tail_side)
    return p["background"] + p["slope"] * (x - fit.x_mean) + p["amplitude"] * profile


def _fit_inputs(
    spec: RateSpectrum,
    mass_u: float,
    options: dict[str, Any],
    voltage_offset_V: float | None = None,
    nu_ref: float | None = None,
    ripple_hw: float | None = None,
) -> dict[str, Any]:
    """Doppler-corrected cells (x relative to nu_ref, MHz) and the fixed model settings."""
    offset = float(options.get("voltage_offset_V", 0.0) if voltage_offset_V is None else voltage_offset_V)
    doppler_kwargs = {
        "charge_e": options.get("charge_e", 1),
        "geometry": options.get("geometry", "collinear"),
        "neutralization": options.get("neutralization", "none"),
        "sodium_collision_branch": options.get("sodium_collision_branch", "forward"),
    }
    voltage = spec.voltage_V + offset
    nu_rest = two_fit.doppler_correct_ghz(
        spec.nu_lab_MHz, mass_u, voltage, doppler_kwargs["charge_e"], doppler_kwargs["geometry"],
        neutralization=doppler_kwargs["neutralization"], sodium_mass_u=options.get("sodium_mass_u", two_fit.SODIUM_MASS_U),
        sodium_collision_branch=doppler_kwargs["sodium_collision_branch"],
    )
    if nu_ref is None:
        weights = spec.counts if spec.counts.sum() > 0 else spec.bunches
        order = np.argsort(nu_rest)
        cum = np.cumsum(weights[order])
        nu_ref = float(nu_rest[order][np.searchsorted(cum, 0.5 * cum[-1])])
    x = nu_rest - nu_ref
    mean_voltage = float(np.average(voltage, weights=spec.bunches))
    if ripple_hw is None:
        ripple_V = float(v2_option(options, "ripple_amplitude_V") or 0.0)
        ripple_hw = rl.ripple_halfwidth_MHz(nu_ref, mass_u, mean_voltage, ripple_V, **doppler_kwargs)
    slope_MHz_per_V, tail_side = doppler_slope(nu_ref, mass_u, mean_voltage, doppler_kwargs)
    tail_model = str(v2_option(options, "tail_model") or "none").lower()
    tail_fraction = v2_option(options, "tail_fraction")
    tail_length_V = v2_option(options, "tail_length_V")
    sigma_V = v2_option(options, "sigma_doppler_V")
    fixed_gamma = options.get("lorentzian_hwhm_MHz")
    exposure_mode = bool(v2_option(options, "exposure_normalization"))
    bin_width = float(options.get("bin_width_MHz") or 20.0)
    if exposure_mode:
        fx, fn = x, spec.counts
        fe = spec.bunches * (spec.live if v2_option(options, "deadtime_correction") else 1.0)
    else:
        # Raw counts per frequency bin (exposure ignored): the legacy data model.
        fx, fn, _, _ = _binned(x, spec.counts, spec.bunches, bin_width)
        fe = np.ones_like(fx)
    return {
        "fx": fx, "fn": fn, "fe": fe, "nu_ref": nu_ref, "ripple_hw": float(ripple_hw),
        "ripple_V": float(v2_option(options, "ripple_amplitude_V") or 0.0),
        "laser_fwhm": float(v2_option(options, "laser_linewidth_fwhm_MHz") or 0.0),
        "laser_shape": str(v2_option(options, "laser_lineshape")),
        "fixed_gamma": None if fixed_gamma in (None, "") else float(fixed_gamma),
        "fit_ripple": bool(v2_option(options, "fit_ripple_amplitude")),
        "exposure_mode": exposure_mode,
        "bin_width": bin_width,
        "slope_MHz_per_V": slope_MHz_per_V,
        "tail_model": tail_model,
        "tail_side": tail_side,
        "fixed_tail_fraction": None if tail_fraction in (None, "") else float(tail_fraction),
        "fixed_tail_length": None if tail_length_V in (None, "") else float(tail_length_V) * slope_MHz_per_V,
        "fixed_sigma": None if sigma_V in (None, "") else float(sigma_V) * slope_MHz_per_V,
        "fit_slope": bool(v2_option(options, "fit_background_slope")),
    }


def doppler_slope(nu_ref_MHz: float, mass_u: float, voltage_V: float, doppler_kwargs: dict[str, Any]) -> tuple[float, int]:
    """Doppler-corrected MHz per volt of beam energy, and the side an energy loss moves a line to.

    An atom that lost energy is resonant where its (slower) Doppler factor puts it;
    corrected with the nominal voltage it lands at nu0 D(V)/D(V - dE): below nu0 when
    D falls with V (collinear), above it when D rises (anticollinear).
    """
    per_V = rl.doppler_log_slope_per_V(mass_u, voltage_V, **doppler_kwargs)
    d_hi = two_fit.doppler_correct_ghz(1.0, mass_u, voltage_V + 1.0, doppler_kwargs["charge_e"], doppler_kwargs["geometry"],
                                       neutralization=doppler_kwargs["neutralization"],
                                       sodium_collision_branch=doppler_kwargs["sodium_collision_branch"])
    d_lo = two_fit.doppler_correct_ghz(1.0, mass_u, voltage_V - 1.0, doppler_kwargs["charge_e"], doppler_kwargs["geometry"],
                                       neutralization=doppler_kwargs["neutralization"],
                                       sodium_collision_branch=doppler_kwargs["sodium_collision_branch"])
    return float(nu_ref_MHz) * per_V, (-1 if float(d_hi) < float(d_lo) else 1)


def _lineshape_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """fit_poisson_lineshape keyword arguments from _fit_inputs."""
    return dict(
        ripple_halfwidth=inputs["ripple_hw"], laser_fwhm_MHz=inputs["laser_fwhm"], laser_lineshape=inputs["laser_shape"],
        fit_ripple=inputs["fit_ripple"], fixed_gamma=inputs["fixed_gamma"], tail_model=inputs["tail_model"],
        tail_side=inputs["tail_side"], fixed_tail_fraction=inputs["fixed_tail_fraction"],
        fixed_tail_length=inputs["fixed_tail_length"], fixed_sigma=inputs["fixed_sigma"], fit_slope=inputs["fit_slope"],
    )


def fit_rate_spectrum(
    spec: RateSpectrum,
    mass_u: float,
    options: dict[str, Any],
    *,
    voltage_offset_V: float | None = None,
) -> dict[str, Any]:
    """Fit one isotope block; returns the keys of quick_isotope_shift._fit_absolute_center."""
    inputs = _fit_inputs(spec, mass_u, options, voltage_offset_V)
    fx, fn, fe, nu_ref = inputs["fx"], inputs["fn"], inputs["fe"], inputs["nu_ref"]
    ripple_V, laser_fwhm, laser_shape = inputs["ripple_V"], inputs["laser_fwhm"], inputs["laser_shape"]
    exposure_mode, bin_width = inputs["exposure_mode"], inputs["bin_width"]
    fit = fit_poisson_lineshape(fx, fn, fe, **_lineshape_kwargs(inputs))

    # Display bins and goodness of fit on them.
    mu_cells = fe * np.clip(model_rate(fit, fx, laser_fwhm_MHz=laser_fwhm, laser_lineshape=laser_shape), 1e-12, None)
    centers, nb, eb, idx = _binned(fx, fn, fe, bin_width)
    mb = np.bincount(idx, weights=mu_cells, minlength=centers.size)
    populated = eb > 0
    dof = max(int(populated.sum()) - fit.n_free, 1)
    pearson = float(np.sum((nb[populated] - mb[populated]) ** 2 / np.clip(mb[populated], 1e-12, None)) / dof)
    rate = np.where(populated, nb / np.where(populated, eb, 1.0), np.nan)
    rate_err = np.where(populated, np.sqrt(np.clip(nb, 1.0, None)) / np.where(populated, eb, 1.0), np.nan)
    model_bins = np.where(populated, mb / np.where(populated, eb, 1.0), np.nan)
    signal = np.where(populated, rate - fit.params["background"] - fit.params["slope"] * (centers - fit.x_mean), -np.inf)
    i_peak = int(np.argmax(np.where(populated & (nb >= 5), signal, -np.inf))) if np.any(populated & (nb >= 5)) else int(np.argmax(nb))
    peak_to_model = float(rate[i_peak] / model_bins[i_peak]) if populated[i_peak] and model_bins[i_peak] > 0 else float("nan")
    x_curve = np.linspace(fx.min(), fx.max(), 1500)
    y_curve = model_rate(fit, x_curve, laser_fwhm_MHz=laser_fwhm, laser_lineshape=laser_shape)

    center_unc = fit.errors["center"]
    if not np.isfinite(center_unc) or center_unc <= 0:
        center_unc = float("nan")
    quality = {
        "analysis_model": "v2",
        "peak_to_model_max": peak_to_model,
        "reduced_chi2": pearson,
        "deviance_per_dof": float(fit.deviance / max(fit.n_data - fit.n_free, 1)),
        "max_bin_center_MHz": float(centers[i_peak]),
        "fit_window_min_MHz": float(fx.min()),
        "fit_window_max_MHz": float(fx.max()),
        "exposure_normalization": exposure_mode,
        "ripple_amplitude_V": ripple_V,
        "ripple_halfwidth_MHz": float(fit.params["ripple_halfwidth"]),
        "ripple_halfwidth_unc_MHz": float(fit.errors["ripple_halfwidth"]),
        "laser_fwhm_MHz": laser_fwhm,
        "laser_lineshape": laser_shape,
        "sigma_doppler_MHz": float(fit.params["sigma_doppler"]),
        "gamma_MHz": float(fit.params["gamma"]),
        "fit_converged": fit.success,
        **{k: v for k, v in spec.diagnostics.items() if k != "echo_windows_ns"},
    }
    if fit.tail_model != "none":
        s = inputs["slope_MHz_per_V"]
        quality.update({
            "tail_model": fit.tail_model,
            "tail_side": fit.tail_side,
            "tail_fixed": inputs["fixed_tail_fraction"] is not None and inputs["fixed_tail_length"] is not None,
            "sigma_fixed": inputs["fixed_sigma"] is not None,
            "tail_fraction": float(fit.params["tail_fraction"]),
            "tail_fraction_unc": float(fit.errors["tail_fraction"]),
            "tail_length_MHz": float(fit.params["tail_length"]),
            "tail_length_unc_MHz": float(fit.errors["tail_length"]),
            "tail_length_V": float(fit.params["tail_length"]) / s,
            "sigma_doppler_V": float(fit.params["sigma_doppler"]) / s,
            "sigma_doppler_unc_MHz": float(fit.errors["sigma_doppler"]),
            "gamma_unc_MHz": float(fit.errors["gamma"]),
            "background_slope_per_GHz": float(fit.params["slope"]) * 1000.0,
            "doppler_slope_MHz_per_V": s,
            "deviance": float(fit.deviance),
            "deviance_symmetric": fit.deviance_symmetric,
            "counting_center_unc_MHz": fit.center_unc_symmetric if fit.center_unc_symmetric else center_unc,
        })
    return {
        "analysis_model": "v2",
        "center_abs_GHz": (nu_ref + fit.params["center"]) / 1000.0,
        "center_fit_unc_GHz": center_unc / 1000.0,
        "nu_ref_GHz": nu_ref / 1000.0,
        "x_GHz": fx / 1000.0,
        "counts": nb,
        "centers_GHz": centers / 1000.0,
        "fit_params": fit.params,
        "fit_errors": fit.errors,
        "x_fit_window_GHz": np.array([fx.min(), fx.max()]) / 1000.0,
        "fit_quality": quality,
        "num_points": spec.num_points,
        "display": {
            "centers_MHz": centers,
            "rate": rate,
            "rate_err": rate_err,
            "model": model_bins,
            "curve_x_MHz": x_curve,
            "curve_y": y_curve,
            "exposure_mode": exposure_mode,
        },
    }


# ---------------------------------------------------------------------------
# Block bootstrap of the centroid uncertainty
# ---------------------------------------------------------------------------

def bootstrap_strata(spec: RateSpectrum, block_bins: int) -> list[list[np.ndarray]]:
    """Stratified blocks: stratum = (file, segment of block_bins laser steps), unit = scan pass.

    Each replica keeps the scan's frequency coverage and redraws, segment by
    segment, which passes supply the data. block_bins <= 0 makes the whole pass
    the block (the default): fitted pass by pass, the line center moves ~10 MHz
    between passes of one scan (Poisson error ~2 MHz), and splicing segments of
    passes with different offsets builds line shapes that never occurred, which
    inflated the spread up to 2.7x over the per-pass scatter.
    """
    if int(block_bins) <= 0:
        segment = np.zeros_like(spec.group_bin)
    else:
        segment = spec.group_bin // int(block_bins)
    strata: dict[tuple[int, int], dict[int, list[int]]] = {}
    for i, (f, s, p) in enumerate(zip(spec.group_file, segment, spec.group_pass)):
        strata.setdefault((int(f), int(s)), {}).setdefault(int(p), []).append(i)
    return [[np.asarray(v) for v in units.values()] for units in strata.values()]


def small_sample_factor(strata: list[list[np.ndarray]]) -> float:
    """sqrt(n/(n-1)) for n units per stratum: resampling n of n understates the variance."""
    n = np.array([len(units) for units in strata if len(units) > 1], dtype=float)
    return float(math.sqrt(np.median(n / (n - 1.0)))) if n.size else 1.0


def bootstrap_centers(
    spec: RateSpectrum,
    mass_u: float,
    options: dict[str, Any],
    nominal: dict[str, Any],
    *,
    n_boot: int,
    block_bins: int,
    seed: int,
    voltage_offset_V: float | None = None,
) -> tuple[np.ndarray, float]:
    """Absolute centers (GHz) of n_boot block-bootstrap replicas, warm-started at the nominal fit."""
    strata = bootstrap_strata(spec, block_bins)
    rng = np.random.default_rng(seed)
    nu_ref = nominal["nu_ref_GHz"] * 1000.0
    ripple_hw = nominal["fit_params"]["ripple_halfwidth"]
    out = np.full(int(n_boot), np.nan)
    for r in range(int(n_boot)):
        index = np.concatenate([units[k] for units in strata for k in rng.integers(0, len(units), len(units))])
        inputs = _fit_inputs(spec.subset(index), mass_u, options, voltage_offset_V, nu_ref=nu_ref, ripple_hw=ripple_hw)
        try:
            fit = fit_poisson_lineshape(inputs["fx"], inputs["fn"], inputs["fe"], **_lineshape_kwargs(inputs),
                                        start=nominal["fit_params"])
        except Exception:  # a failed replica is dropped, and counted
            continue
        out[r] = (nu_ref + fit.params["center"]) / 1000.0
    return out, small_sample_factor(strata)


def apply_bootstrap(
    result: dict[str, Any],
    spec: RateSpectrum,
    mass_u: float,
    options: dict[str, Any],
    voltage_offset_V: float | None = None,
) -> dict[str, Any]:
    """Replace a v2 result's Fisher center uncertainty by its block-bootstrap spread.

    The spread is the standard deviation of the replicas within 5 robust sigma of
    their median (a failed or runaway refit cannot inflate it), times the
    small-sample factor, floored at the Fisher value (with 2-3 passes the replica
    spread can fall below the counting error by chance). The Fisher value stays in
    fit_quality.
    """
    n_boot = int(v2_option(options, "bootstrap_replicas") or 0)
    if n_boot <= 0 or result.get("analysis_model") != "v2":
        return result
    block = int(v2_option(options, "bootstrap_block_bins"))
    seed = (int(v2_option(options, "bootstrap_seed")) ^ zlib.crc32(np.ascontiguousarray(spec.counts).tobytes())) & 0x7FFFFFFF
    centers, factor = bootstrap_centers(spec, mass_u, options, result, n_boot=n_boot, block_bins=block,
                                        seed=seed, voltage_offset_V=voltage_offset_V)
    good = centers[np.isfinite(centers)]
    if good.size < max(20, n_boot // 4):
        result["fit_quality"]["bootstrap_error"] = f"only {good.size} of {n_boot} replicas converged"
        return result
    median = float(np.median(good))
    robust = 1.4826 * float(np.median(np.abs(good - median)))
    kept = good[np.abs(good - median) <= 5.0 * robust] if robust > 0 else good
    spread = float(np.std(kept, ddof=1)) * factor
    fisher = float(result["center_fit_unc_GHz"])
    quality = result["fit_quality"]
    quality.update({
        "fisher_center_unc_MHz": fisher * 1000.0,
        "bootstrap_center_unc_MHz": spread * 1000.0,
        "bootstrap_replicas": n_boot,
        "bootstrap_used": int(kept.size),
        "bootstrap_block_bins": block,
        "bootstrap_units_per_stratum": int(np.median([len(u) for u in bootstrap_strata(spec, block)])),
        "bootstrap_small_sample_factor": factor,
    })
    result["center_fit_unc_GHz"] = max(spread, fisher)
    return result


def _warm_center_MHz(spec: RateSpectrum, mass_u: float, options: dict[str, Any], nominal: dict[str, Any],
                     index: np.ndarray, voltage_offset_V: float | None = None) -> tuple[float, dict[str, Any]]:
    """Center (MHz) of a resampled spectrum, warm-started at the nominal fit; also its shape (volts)."""
    inputs = _fit_inputs(spec.subset(index), mass_u, options, voltage_offset_V, nu_ref=nominal["nu_ref_GHz"] * 1000.0,
                         ripple_hw=nominal["fit_params"]["ripple_halfwidth"])
    fit = fit_poisson_lineshape(inputs["fx"], inputs["fn"], inputs["fe"], **_lineshape_kwargs(inputs), start=nominal["fit_params"])
    s = inputs["slope_MHz_per_V"]
    shape = {"fit_quality": {"tail_fraction": fit.params["tail_fraction"], "tail_length_V": fit.params["tail_length"] / s,
                             "sigma_doppler_V": fit.params["sigma_doppler"] / s}}
    return nominal["nu_ref_GHz"] * 1000.0 + fit.params["center"], shape


def pair_bootstrap_shift(
    references: list[tuple[RateSpectrum, float, dict[str, Any]]],
    comparison: tuple[RateSpectrum, float, dict[str, Any]],
    options: dict[str, Any],
    *,
    weights=None,
    voltage_offset_V: float | None = None,
) -> dict[str, Any] | None:
    """Bootstrap of the shift comparison - sum(w * reference) that keeps the shape transfer inside.

    Every replica resamples the scan passes of all spectra, refits the reference(s)
    with the free tail, hands that replica's tail to the comparison fit and takes
    the difference. The tail is common to both isotopes, so its uncertainty largely
    cancels in the shift; resampling the isotopes separately (apply_bootstrap)
    breaks that correlation and overstates the shift error severalfold.
    Entries are (spectrum, mass, nominal v2 result). Returns None when disabled.
    """
    n_boot = int(v2_option(options, "bootstrap_replicas") or 0)
    if n_boot <= 0:
        return None
    w = np.ones(len(references)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    block = int(v2_option(options, "bootstrap_block_bins"))
    specs = [ref[0] for ref in references] + [comparison[0]]
    strata = [bootstrap_strata(spec, block) for spec in specs]
    seed = int(v2_option(options, "bootstrap_seed"))
    for spec in specs:
        seed ^= zlib.crc32(np.ascontiguousarray(spec.counts).tobytes())
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    shifts = np.full(n_boot, np.nan)
    for r in range(n_boot):
        index = [np.concatenate([units[k] for units in st for k in rng.integers(0, len(units), len(units))]) for st in strata]
        try:
            ref_fits = [_warm_center_MHz(spec, mass, options, nominal, idx, voltage_offset_V)
                        for (spec, mass, nominal), idx in zip(references, index[:-1])]
            shaped = transferred_shape_options(options, [shape for _, shape in ref_fits], w)
            spec, mass, nominal = comparison
            center, _ = _warm_center_MHz(spec, mass, shaped, nominal, index[-1], voltage_offset_V)
        except Exception:  # a failed replica is dropped, and counted
            continue
        shifts[r] = center - float(np.dot(w, [c for c, _ in ref_fits]))
    good = shifts[np.isfinite(shifts)]
    if good.size < max(20, n_boot // 4):
        return {"error": f"only {good.size} of {n_boot} replicas converged"}
    median = float(np.median(good))
    robust = 1.4826 * float(np.median(np.abs(good - median)))
    kept = good[np.abs(good - median) <= 5.0 * robust] if robust > 0 else good
    factor = float(np.sqrt(np.median([small_sample_factor(st) ** 2 for st in strata])))
    return {"shift_unc_MHz": float(np.std(kept, ddof=1)) * factor, "replicas": n_boot, "used": int(kept.size),
            "small_sample_factor": factor}


# ---------------------------------------------------------------------------
# Two-isotope comparison (legacy plot_two_isotopes_fit result shape)
# ---------------------------------------------------------------------------

def plot_rate_fit(ax, result: dict[str, Any], *, reference_GHz: float, label: str, color: str) -> None:
    disp = result["display"]
    shift_MHz = (result["nu_ref_GHz"] - reference_GHz) * 1000.0
    ok = np.isfinite(disp["rate"])
    ax.errorbar(disp["centers_MHz"][ok] + shift_MHz, disp["rate"][ok], yerr=disp["rate_err"][ok], fmt="o", ms=4,
                capsize=2, color=color, ecolor="black", label=label)
    ax.plot(disp["curve_x_MHz"] + shift_MHz, disp["curve_y"], color=color, lw=2)
    center_MHz = (result["center_abs_GHz"] - reference_GHz) * 1000.0
    q = result["fit_quality"]
    ax.axvline(center_MHz, color=color, linestyle="--",
               label=f"center = {center_MHz:.1f} +/- {result['center_fit_unc_GHz'] * 1000.0:.1f} MHz")
    ax.set_ylabel("Ions per bunch" if disp["exposure_mode"] else "Counts", fontweight="bold")
    ax.set_title(
        f"{label}: chi2/dof {q['reduced_chi2']:.2f}, echoes {q.get('echo_fraction', 0.0):.1%}, "
        f"min live {q.get('min_live_fraction', 1.0):.2f}, ripple a {q['ripple_halfwidth_MHz']:.0f} MHz",
        fontweight="bold",
    )
    style_axes(ax)
    ax.legend()


SHAPE_TRANSFERS = ("none", "tail", "tail+sigma")


def counting_unc_MHz(result: dict[str, Any]) -> float:
    """Counting (Fisher) center error of a v2 result, with any energy-loss tail held fixed."""
    q = result["fit_quality"]
    return float(q.get("counting_center_unc_MHz", q.get("fisher_center_unc_MHz", result["center_fit_unc_GHz"] * 1000.0)))


def shape_transfer_mode(options: dict[str, Any]) -> str:
    """Active shape transfer ("none" unless an energy-loss tail is modeled)."""
    mode = str(v2_option(options, "shape_transfer") or "none").lower()
    if mode not in SHAPE_TRANSFERS:
        raise ValueError(f"shape_transfer must be one of {SHAPE_TRANSFERS}, got {mode!r}")
    if str(v2_option(options, "tail_model") or "none").lower() == "none":
        return "none"
    return mode


def reference_is_first(label1: str, label2: str, spec1: RateSpectrum, spec2: RateSpectrum) -> bool:
    """The shape reference of a pair: 32S if present, else the spectrum with more ions."""
    if "32S" in (label1, label2):
        return label1 == "32S"
    return spec1.num_points >= spec2.num_points


def transferred_shape_options(options: dict[str, Any], references: list[dict[str, Any]], weights=None) -> dict[str, Any]:
    """Options with the reference fits' tail (and, for "tail+sigma", Gaussian width) fixed, in volts.

    The energy-loss tail and the energy spread are properties of the beam, the same
    in volts for both isotopes of a pair; the Doppler slope of each isotope turns
    them into MHz. Several references (the 32S scans bracketing a 34S scan) are
    averaged with ``weights``.
    """
    mode = shape_transfer_mode(options)
    if mode == "none":
        return options
    w = np.ones(len(references)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    quality = [ref["fit_quality"] for ref in references]
    out = dict(options)
    out["tail_fraction"] = float(np.dot(w, [q["tail_fraction"] for q in quality]))
    out["tail_length_V"] = float(np.dot(w, [q["tail_length_V"] for q in quality]))
    if mode == "tail+sigma":
        out["sigma_doppler_V"] = float(np.dot(w, [q["sigma_doppler_V"] for q in quality]))
    return out


def two_isotope_rate_fit(
    spec1: RateSpectrum,
    spec2: RateSpectrum,
    *,
    mass1_u: float,
    mass2_u: float,
    label1: str,
    label2: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    apply_publication_style()
    offset = float(options.get("voltage_offset_V", 0.0))
    options1 = options2 = options
    if shape_transfer_mode(options) == "none":
        r1 = apply_bootstrap(fit_rate_spectrum(spec1, mass1_u, options), spec1, mass1_u, options)
        r2 = apply_bootstrap(fit_rate_spectrum(spec2, mass2_u, options), spec2, mass2_u, options)
    elif reference_is_first(label1, label2, spec1, spec2):
        r1 = apply_bootstrap(fit_rate_spectrum(spec1, mass1_u, options), spec1, mass1_u, options)
        options2 = transferred_shape_options(options, [r1])
        r2 = apply_bootstrap(fit_rate_spectrum(spec2, mass2_u, options2), spec2, mass2_u, options2)
    else:
        r2 = apply_bootstrap(fit_rate_spectrum(spec2, mass2_u, options), spec2, mass2_u, options)
        options1 = transferred_shape_options(options, [r2])
        r1 = apply_bootstrap(fit_rate_spectrum(spec1, mass1_u, options1), spec1, mass1_u, options1)
    nu0 = 0.5 * (r1["nu_ref_GHz"] + r2["nu_ref_GHz"])
    center1 = r1["center_abs_GHz"] - nu0
    center2 = r2["center_abs_GHz"] - nu0
    shift = center2 - center1
    fit_unc = math.hypot(r1["center_fit_unc_GHz"], r2["center_fit_unc_GHz"])
    pair_quality = None
    if shape_transfer_mode(options) != "none":
        # The shift error from resampling both isotopes together (tail cancels).
        first = reference_is_first(label1, label2, spec1, spec2)
        ref, cmp_ = ((spec1, mass1_u, r1), (spec2, mass2_u, r2)) if first else ((spec2, mass2_u, r2), (spec1, mass1_u, r1))
        pair_quality = pair_bootstrap_shift([ref], cmp_, options)
        if pair_quality and "shift_unc_MHz" in pair_quality:
            # Floor: the counting error with the tail held common (the free-tail core error
            # of the reference carries the core/tail degeneracy that cancels in the pair).
            fisher = [counting_unc_MHz(q) for q in (r1, r2)]
            fit_unc = max(pair_quality["shift_unc_MHz"], math.hypot(*fisher)) / 1000.0

    unc_V = float(options.get("beam_voltage_unc_V", 0.0) or 0.0)
    d1 = d2 = dshift = 0.0
    if unc_V > 0:
        shifted = {}
        for sign in (1.0, -1.0):
            a = fit_rate_spectrum(spec1, mass1_u, options1, voltage_offset_V=offset + sign * unc_V)["center_abs_GHz"]
            b = fit_rate_spectrum(spec2, mass2_u, options2, voltage_offset_V=offset + sign * unc_V)["center_abs_GHz"]
            shifted[sign] = (a, b)
        d1 = abs(shifted[1.0][0] - shifted[-1.0][0]) / 2.0
        d2 = abs(shifted[1.0][1] - shifted[-1.0][1]) / 2.0
        dshift = abs((shifted[1.0][1] - shifted[1.0][0]) - (shifted[-1.0][1] - shifted[-1.0][0])) / 2.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    plot_rate_fit(axes[0], r1, reference_GHz=nu0, label=label1, color="C0")
    plot_rate_fit(axes[1], r2, reference_GHz=nu0, label=label2, color="C1")
    axes[1].set_xlabel("Corrected frequency relative to nu0 (MHz)", fontweight="bold")
    fig.suptitle(f"{label2}-{label1} (analysis v2): shift = {shift * 1000.0:.1f} +/- {fit_unc * 1000.0:.1f} MHz",
                 fontweight="bold")
    plt.tight_layout()

    return {
        "analysis_model": "v2",
        "nu0_GHz": nu0,
        "center1_GHz": float(center1),
        "center1_fit_unc_GHz": float(r1["center_fit_unc_GHz"]),
        "center1_voltage_unc_GHz": float(d1),
        "center1_total_unc_GHz": float(math.hypot(r1["center_fit_unc_GHz"], d1)),
        "center2_GHz": float(center2),
        "center2_fit_unc_GHz": float(r2["center_fit_unc_GHz"]),
        "center2_voltage_unc_GHz": float(d2),
        "center2_total_unc_GHz": float(math.hypot(r2["center_fit_unc_GHz"], d2)),
        "isotope_shift_GHz": float(shift),
        "isotope_shift_fit_unc_GHz": float(fit_unc),
        "isotope_shift_voltage_unc_GHz": float(dshift),
        "isotope_shift_total_unc_GHz": float(math.hypot(fit_unc, dshift)),
        "num_points_1": spec1.num_points,
        "num_points_2": spec2.num_points,
        "fit_quality": {label1: r1["fit_quality"], label2: r2["fit_quality"],
                        **({"pair_bootstrap": pair_quality} if pair_quality else {})},
        "results": {label1: r1, label2: r2},
    }
