"""
Simulated resonance-ionization spectra for the FRIB neutron-deficient sulfur
proposal (Fig. 5 placeholder).

This mirrors the kind of "expected spectrum" figure used in the earlier 22Si
RISE proposal: build the expected lineshape vs. laser detuning, fold in the
beam yield / efficiency / background / measurement time to get expected counts
per frequency step, Poisson-sample to make synthetic "Data", overlay the fit,
and read off the statistical precision on the isotope shift (even-even) or on
the hyperfine A/B constants (odd-A).

Two cases are provided:
    * even-even (spin-0+):  a single Voigt line   -> isotope-shift precision
    * odd-A (hyperfine):    sum of Voigt lines at the hyperfine-component
                            positions of the 4s 5S2 -> 6p 5P2 transition
                            (J_lower = J_upper = 2) -> A/B precision

Counting convention (peak-rate, the physically transparent one)
---------------------------------------------------------------
    detected_peak_rate = stopped_beam_rate_pps * total_efficiency      [counts/s]
The laser scan visits ``n_steps`` frequency points; over the full run each
point is dwelled for a total time ``dwell = measurement_time_s / n_steps``.
Expected counts in step i:
    lambda_i = detected_peak_rate * profile_norm(nu_i) * dwell
               + background_rate_cps * dwell
where profile_norm peaks at 1.0 on the strongest resonance. Counts are then
drawn from Poisson(lambda_i).

``total_efficiency`` here is the *on-resonance* overall detection efficiency
(neutralization x resonant ionization x BECOLA transmission x particle
detection). Swap in real FRIB yield estimates via the ISOTOPES table below.

Run directly to regenerate the two example figures:
    python proposal_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.physics.wigner import wigner_6j

# Shared analysis modules live at the repo root; make them importable no matter the
# working directory (this module now lives under hfs_gui/).
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_style import apply_publication_style, style_axes
from isotope_shift_analysis import (
    voigt, _voigt_fwhm, B_HVD2,
    apply_tof_gate, _lab_frequency_and_voltage, doppler_correct_ghz,
)

# Atomic masses (u) for the Doppler correction of real commissioning scans.
MASS_U = {"32S": 31.97207117, "34S": 33.96786700}


# --------------------------------------------------------------------------
# Hyperfine structure for an electric-dipole transition J_l -> J_u
# --------------------------------------------------------------------------
def _hfs_level_shift(F, I, J, A, B):
    """Hyperfine energy shift of level (I, J) at total angular momentum F.

    E = A*K/2 + B * [ (3/2)K(K+1) - 2 I(I+1) J(J+1) ]
                    / [ 2 I (2I-1) J (2J-1) ]
    with K = F(F+1) - I(I+1) - J(J+1). The B term vanishes unless both
    I >= 1 and J >= 1 (no quadrupole interaction otherwise).
    """
    I = float(I); J = float(J)
    K = F * (F + 1.0) - I * (I + 1.0) - J * (J + 1.0)
    shift = 0.5 * A * K
    denom = 2.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0)
    if B != 0.0 and abs(denom) > 1e-9:
        casimir = (1.5 * K * (K + 1.0) - 2.0 * I * (I + 1.0) * J * (J + 1.0)) / denom
        shift += B * casimir
    return shift


def _f_values(I, J):
    Fmin = abs(I - J)
    Fmax = I + J
    n = int(round(Fmax - Fmin)) + 1
    return [Fmin + k for k in range(n)]


def hfs_components(I, J_lower, J_upper):
    """Return the allowed (F_lower, F_upper, relative_intensity) components.

    Relative line strengths for an electric-dipole transition:
        S ∝ (2F_l+1)(2F_u+1) * { J_u  F_u  I ;  F_l  J_l  1 }^2
    Selection rules (Delta F = 0, +-1, no 0->0) are enforced automatically by
    the 6j symbol vanishing. Intensities are normalised so the strongest
    component = 1.
    """
    comps = []
    for Fl in _f_values(I, J_lower):
        for Fu in _f_values(I, J_upper):
            if abs(Fl - Fu) > 1:
                continue
            sixj = float(wigner_6j(J_upper, Fu, I, Fl, J_lower, 1))
            if sixj == 0.0:
                continue
            strength = (2 * Fl + 1) * (2 * Fu + 1) * sixj ** 2
            if strength > 0:
                comps.append((Fl, Fu, strength))
    smax = max(c[2] for c in comps)
    return [(Fl, Fu, s / smax) for (Fl, Fu, s) in comps]


def hfs_pattern(I, J_lower, J_upper, A_lower, B_lower, A_upper, B_upper,
                centroid=0.0):
    """Component (offset_MHz, relative_intensity) list for given A/B constants."""
    out = []
    for Fl, Fu, rel in hfs_components(I, J_lower, J_upper):
        nu = (centroid
              + _hfs_level_shift(Fu, I, J_upper, A_upper, B_upper)
              - _hfs_level_shift(Fl, I, J_lower, A_lower, B_lower))
        out.append((nu, rel))
    return out


# --------------------------------------------------------------------------
# Spectrum simulation
# --------------------------------------------------------------------------
def _components_for(spec):
    """Build (offset, rel_intensity) components for an isotope spec dict."""
    if spec["I"] == 0:
        return [(0.0, 1.0)]
    return hfs_pattern(
        spec["I"], spec["J_lower"], spec["J_upper"],
        spec.get("A_lower", 0.0), spec.get("B_lower", 0.0),
        spec.get("A_upper", 0.0), spec.get("B_upper", 0.0),
        centroid=0.0,
    )


def expected_counts(x_MHz, components, detected_peak_rate, dwell_s,
                    background_cps, sigma_g, gamma_l):
    """Expected counts per frequency step (no noise)."""
    x = np.asarray(x_MHz, dtype=float)
    profile = np.zeros_like(x)
    for nu, rel in components:
        # unit-peak Voigt at nu, scaled by relative intensity
        profile += rel * voigt(x, 1.0, nu, sigma_g, gamma_l, 0.0)
    return detected_peak_rate * dwell_s * profile + background_cps * dwell_s


def simulate_spectrum(spec, stopped_beam_rate_pps, total_efficiency,
                      background_cps, measurement_time_s,
                      scan_range_MHz=(-400.0, 400.0), n_steps=160,
                      fwhm_MHz=40.0, lorentz_fraction=0.5, seed=0):
    """Simulate one Poisson-sampled scan.

    Returns dict with x, counts, expected, and the derived scalars.
    """
    lo, hi = scan_range_MHz
    x = np.linspace(lo, hi, n_steps)
    dwell = measurement_time_s / n_steps
    detected_peak_rate = stopped_beam_rate_pps * total_efficiency

    # split requested FWHM into Gaussian + Lorentzian parts (Voigt FWHM is a
    # known function of sigma_g, gamma_l; invert approximately via scaling).
    gamma_l = lorentz_fraction * fwhm_MHz / 2.0
    sigma_g = (1.0 - lorentz_fraction) * fwhm_MHz / 2.3548
    # rescale so the realised Voigt FWHM matches the requested value
    scale = fwhm_MHz / _voigt_fwhm(sigma_g, gamma_l)
    sigma_g *= scale
    gamma_l *= scale

    comps = _components_for(spec)
    lam = expected_counts(x, comps, detected_peak_rate, dwell, background_cps,
                          sigma_g, gamma_l)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam).astype(float)

    return {
        "x": x, "counts": counts, "expected": lam,
        "components": comps, "dwell_s": dwell,
        "detected_peak_rate_cps": detected_peak_rate,
        "sigma_g": sigma_g, "gamma_l": gamma_l, "fwhm_MHz": fwhm_MHz,
        "total_counts": float(counts.sum()),
        "peak_counts": float(counts.max()),
        "spec": spec,
        "stopped_beam_rate_pps": stopped_beam_rate_pps,
        "total_efficiency": total_efficiency,
        "background_cps": background_cps,
        "measurement_time_s": measurement_time_s,
    }


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_even_even(x, counts):
    """Single-Voigt fit; returns (center, center_err, popt).

    popt = (amplitude, center, sigma_g, gamma_l, background) -- same order as
    isotope_shift_analysis.voigt, so the result can be passed straight to it.
    """
    x = np.asarray(x, dtype=float)
    counts = np.asarray(counts, dtype=float)
    yerr = np.sqrt(np.clip(counts, 1.0, None))

    bkg0 = max(np.median(counts), 0.1)
    amp0 = max(counts.max() - bkg0, 1.0)
    center0 = x[int(np.argmax(counts))]
    width0 = max((x.max() - x.min()) / 20.0, 1e-3)
    p0 = [amp0, center0, width0, 0.5 * width0, bkg0]
    lower = [0.0, x.min(), 1e-3, 1e-3, 0.0]
    upper = [np.inf, x.max(), x.max() - x.min(), x.max() - x.min(),
             max(counts.max(), 1.0)]
    popt, pcov = curve_fit(voigt, x, counts, p0=p0, sigma=yerr,
                           absolute_sigma=True, bounds=(lower, upper),
                           maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    return popt[1], perr[1], popt


def _hfs_model_builder(spec):
    """Return (model_fn, p0_template) for curve_fit of an odd-A spectrum.

    Free params: centroid, A_lower, B_lower, A_upper, B_upper, amplitude,
                  sigma_g, gamma_l, background.
    Relative component intensities are fixed (depend only on I, J, F).
    """
    I = spec["I"]; Jl = spec["J_lower"]; Ju = spec["J_upper"]
    comps = hfs_components(I, Jl, Ju)            # (Fl, Fu, rel) fixed
    has_B = (I >= 1.0) and (Jl >= 1.0 or Ju >= 1.0)

    def model(x, centroid, A_lower, B_lower, A_upper, B_upper,
              amplitude, sigma_g, gamma_l, background):
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        for Fl, Fu, rel in comps:
            nu = (centroid
                  + _hfs_level_shift(Fu, I, Ju, A_upper, B_upper)
                  - _hfs_level_shift(Fl, I, Jl, A_lower, B_lower))
            y += rel * voigt(x, 1.0, nu, sigma_g, gamma_l, 0.0)
        return amplitude * y + background

    return model, has_B


def fit_hfs(sim):
    """Fit a simulated odd-A spectrum; returns dict of fitted params + errors."""
    spec = sim["spec"]
    x = sim["x"]; counts = sim["counts"]
    model, has_B = _hfs_model_builder(spec)

    yerr = np.sqrt(np.clip(counts, 1.0, None))
    amp0 = max(counts.max() - np.median(counts), 1.0)
    span = float(x.max() - x.min())
    p0 = [0.0,
          spec.get("A_lower", 100.0), spec.get("B_lower", 0.0),
          spec.get("A_upper", 100.0), spec.get("B_upper", 0.0),
          amp0, sim["sigma_g"], sim["gamma_l"], max(np.median(counts), 0.1)]
    # Physical bounds keep widths positive and avoid the degenerate
    # sigma_g/gamma_l trade-off that leaves the covariance unestimable.
    lower = [x.min(), -2000.0, -2000.0, -2000.0, -2000.0,
             0.0, 1e-3, 1e-3, 0.0]
    upper = [x.max(), 2000.0, 2000.0, 2000.0, 2000.0,
             np.inf, span, span, max(counts.max(), 1.0)]

    def _clip(p):
        return [min(max(v, lo), hi) for v, lo, hi in zip(p, lower, upper)]

    # If no quadrupole interaction, hold B at zero to avoid a degenerate fit.
    if not has_B:
        fixed = {2: 0.0, 4: 0.0}
        free_idx = [i for i in range(len(p0)) if i not in fixed]

        def model_reduced(x, *free):
            full = list(p0)
            for j, i in enumerate(free_idx):
                full[i] = free[j]
            for i, v in fixed.items():
                full[i] = v
            return model(x, *full)

        p0r = _clip(p0)
        p0r = [p0r[i] for i in free_idx]
        lo_r = [lower[i] for i in free_idx]
        hi_r = [upper[i] for i in free_idx]
        popt_r, pcov_r = curve_fit(model_reduced, x, counts, p0=p0r,
                                   sigma=yerr, absolute_sigma=True,
                                   bounds=(lo_r, hi_r), maxfev=40000)
        perr_r = np.sqrt(np.diag(pcov_r))
        popt = list(p0); perr = [0.0] * len(p0)
        for j, i in enumerate(free_idx):
            popt[i] = popt_r[j]; perr[i] = perr_r[j]
    else:
        popt, pcov = curve_fit(model, x, counts, p0=_clip(p0), sigma=yerr,
                               absolute_sigma=True, bounds=(lower, upper),
                               maxfev=40000)
        perr = np.sqrt(np.diag(pcov))

    names = ["centroid", "A_lower", "B_lower", "A_upper", "B_upper",
             "amplitude", "sigma_g", "gamma_l", "background"]
    return {n: (float(popt[i]), float(perr[i])) for i, n in enumerate(names)}, model, popt


# --------------------------------------------------------------------------
# Plotting (22Si style)
# --------------------------------------------------------------------------
def plot_simulated_spectrum(sim, ax=None, title=None, data_label="Data",
                            fit_label="Fit", color="C0"):
    apply_publication_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    x = sim["x"]; counts = sim["counts"]
    yerr = np.sqrt(np.clip(counts, 0.0, None))
    yerr[yerr == 0] = 1.0

    xf = np.linspace(x.min(), x.max(), 2000)
    spec = sim["spec"]
    if spec["I"] == 0:
        center, center_err, popt = fit_even_even(x, counts)
        yf = voigt(xf, *popt)
        fit_info = f"centroid: {center:.1f} ± {center_err:.1f} MHz"
    else:
        params, model, popt = fit_hfs(sim)
        yf = model(xf, *popt)
        a = params["A_lower"]
        fit_info = f"A(5S) = {a[0]:.1f} ± {a[1]:.1f} MHz"

    ax.plot(xf, yf, color="C1", lw=2, label=fit_label, zorder=3)
    ax.errorbar(x, counts, yerr=yerr, fmt="o", ms=4.5, color=color,
                ecolor="black", elinewidth=0.9, capsize=2.0, capthick=0.9,
                label=data_label, zorder=2)
    ax.set_xlabel("Relative frequency (MHz)", fontweight="bold")
    ax.set_ylabel("Counts", fontweight="bold")
    if title:
        ax.text(0.04, 0.92, title, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")
    style_axes(ax)
    ax.legend(loc="upper right")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    return fig, ax, fit_info


# --------------------------------------------------------------------------
# Calibration from a real commissioning scan (lineshape only)
# --------------------------------------------------------------------------
def calibrate_from_scan(file, isotope="32S", wn_col="wavemeter_wn1",
                        tof_gate_us=(4.25, 5.5), tof_col="tof",
                        bin_width_MHz=10.0, frequency_multiplier=2.0,
                        beam_voltage_V=10000.0, voltage_col="voltage",
                        voltage_multiplier=B_HVD2, geometry="collinear"):
    """Fit a real Doppler-corrected scan and return its lineshape parameters.

    Returns a dict with the realised FWHM (MHz), the Gaussian/Lorentzian split
    needed to reproduce it, and the background-to-peak ratio -- everything the
    simulator needs to make the synthetic peaks look like the real data. The
    absolute yield/efficiency is NOT inferred here (it needs the commissioning
    beam current); feed those in separately.
    """
    dat = np.genfromtxt(file, delimiter=",", names=True, dtype=None,
                        encoding="utf-8")
    dat = apply_tof_gate(dat, tof_gate_us=tof_gate_us, tof_col=tof_col)

    nu_lab, voltage_V, _src = _lab_frequency_and_voltage(
        dat, wn_col, frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V, voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier, use_voltage_column=True)
    mass_u = MASS_U.get(isotope, 31.97207117)
    nu_rest = doppler_correct_ghz(nu_lab, mass_u, voltage_V, 1, geometry)

    x_MHz = (nu_rest - np.median(nu_rest)) * 1000.0
    edges = np.arange(x_MHz.min(), x_MHz.max() + bin_width_MHz, bin_width_MHz)
    counts, edges = np.histogram(x_MHz, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    center, center_err, popt = fit_even_even(centers, counts)
    amp, _c, sigma_g, gamma_l, bkg = popt
    fwhm = _voigt_fwhm(sigma_g, gamma_l)
    lorentz_fraction = (2.0 * gamma_l) / max(2.0 * gamma_l + 2.3548 * sigma_g, 1e-9)
    return {
        "isotope": isotope, "file": file,
        "fwhm_MHz": float(fwhm),
        "sigma_g_MHz": float(sigma_g), "gamma_l_MHz": float(gamma_l),
        "lorentz_fraction": float(np.clip(lorentz_fraction, 0.0, 1.0)),
        "peak_counts": float(amp + bkg), "background_counts": float(bkg),
        "background_fraction": float(bkg / max(amp + bkg, 1e-9)),
        "center_MHz": float(center), "center_err_MHz": float(center_err),
        "n_events": int(dat.size),
        "centers_MHz": centers, "counts": counts, "popt": popt,
    }


# --------------------------------------------------------------------------
# Monte-Carlo precision vs. measurement time
# --------------------------------------------------------------------------
def precision_vs_time(spec, stopped_beam_rate_pps, total_efficiency,
                      background_cps, times_s, n_rep=150,
                      observable="centroid", **sim_kwargs):
    """Std of the fitted observable across n_rep Poisson realisations per time.

    observable: 'centroid' (isotope shift) or 'A_lower' (hyperfine constant).
    Returns (times_s, sigma_MHz).
    """
    sigmas = []
    for t in times_s:
        vals = []
        for r in range(n_rep):
            sim = simulate_spectrum(spec, stopped_beam_rate_pps,
                                    total_efficiency, background_cps, t,
                                    seed=1000 * int(t) + r, **sim_kwargs)
            try:
                if spec["I"] == 0:
                    c, _e, _p = fit_even_even(sim["x"], sim["counts"])
                    vals.append(c)
                else:
                    params, _m, _p = fit_hfs(sim)
                    vals.append(params[observable][0])
            except Exception:
                continue
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        sigmas.append(np.std(vals) if vals.size > 2 else np.nan)
    return np.asarray(times_s, dtype=float), np.asarray(sigmas, dtype=float)


# --------------------------------------------------------------------------
# Isotope assumptions  --  EDIT THESE with FRIB yields / your atomic estimates
# --------------------------------------------------------------------------
HOUR = 3600.0

# 4s 5S2 -> 6p 5P2 transition: both levels have J = 2.
J_LOWER = 2
J_UPPER = 2

# A/B hyperfine constants below are PLACEHOLDERS chosen only to make the
# example figures; replace with your atomic-structure estimates.
ISOTOPES = {
    # 27S (5/2+) -- hyperfine (IS + HFS)
    "27S": dict(I=2.5, J_lower=J_LOWER, J_upper=J_UPPER,
                A_lower=140.0, B_lower=-25.0, A_upper=52.0, B_upper=-10.0),
    # 28S (0+) -- even-even
    "28S": dict(I=0, J_lower=J_LOWER, J_upper=J_UPPER),
    # 29S (5/2+) -- hyperfine
    "29S": dict(I=2.5, J_lower=J_LOWER, J_upper=J_UPPER,
                A_lower=120.0, B_lower=-30.0, A_upper=45.0, B_upper=-12.0),
    # 30S (0+) -- even-even
    "30S": dict(I=0, J_lower=J_LOWER, J_upper=J_UPPER),
    # 31S (1/2+) -- magnetic dipole only (I=1/2 -> no quadrupole, B=0)
    "31S": dict(I=0.5, J_lower=J_LOWER, J_upper=J_UPPER,
                A_lower=350.0, A_upper=130.0),
    # 32S (0+) -- stable reference
    "32S": dict(I=0, J_lower=J_LOWER, J_upper=J_UPPER),
}

# Ground-state metadata for the beamtime table (Table 2 of the proposal).
ISOTOPE_META = {
    "27S": dict(Jpi="(5/2)+", half_life="16.3 ms", goal="IS/HFS"),
    "28S": dict(Jpi="0+",     half_life="125 ms",  goal="IS"),
    "29S": dict(Jpi="(5/2)+", half_life="188 ms",  goal="IS/HFS"),
    "30S": dict(Jpi="0+",     half_life="1.18 s",  goal="IS"),
    "31S": dict(Jpi="1/2+",   half_life="2.55 s",  goal="IS/HFS"),
    "32S": dict(Jpi="0+",     half_life="stable",  goal="reference"),
}


def auto_scan_range(spec, fwhm_MHz, pad_fwhm=3.0):
    """Symmetric scan range (MHz) that comfortably covers all components."""
    offs = [nu for nu, _rel in _components_for(spec)]
    half = max(abs(min(offs)), abs(max(offs))) + pad_fwhm * fwhm_MHz
    return (-half, half)


def auto_n_steps(scan_range_MHz, fwhm_MHz, pts_per_fwhm=4.0, minimum=120):
    span = scan_range_MHz[1] - scan_range_MHz[0]
    return int(max(minimum, np.ceil(span / (fwhm_MHz / pts_per_fwhm))))


# --------------------------------------------------------------------------
# Beamtime estimation (Table 2)
# --------------------------------------------------------------------------
def required_hours(spec, stopped_beam_rate_pps, total_efficiency,
                   background_cps, target_MHz, observable="centroid",
                   ref_time_h=24.0, n_rep=60, fwhm_MHz=60.0,
                   lorentz_fraction=0.3, scan_range_MHz=None, n_steps=None):
    """Hours of beam needed to reach ``target_MHz`` on ``observable``.

    Runs one Monte-Carlo point at ``ref_time_h`` to get sigma_ref, then uses
    the Poisson scaling sigma ~ t^-1/2  ->  t_req = t_ref (sigma_ref/target)^2.
    Returns (hours, sigma_ref_MHz).
    """
    if scan_range_MHz is None:
        scan_range_MHz = auto_scan_range(spec, fwhm_MHz)
    if n_steps is None:
        n_steps = auto_n_steps(scan_range_MHz, fwhm_MHz)
    _t, sig = precision_vs_time(
        spec, stopped_beam_rate_pps, total_efficiency, background_cps,
        [ref_time_h * HOUR], n_rep=n_rep, observable=observable,
        scan_range_MHz=scan_range_MHz, n_steps=n_steps,
        fwhm_MHz=fwhm_MHz, lorentz_fraction=lorentz_fraction)
    sigma_ref = float(sig[0])
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        return np.nan, sigma_ref
    return ref_time_h * (sigma_ref / target_MHz) ** 2, sigma_ref


def beamtime_table(rates_pps, total_efficiency, background_cps,
                   target_IS_MHz=3.0, target_A_MHz=5.0, isotopes=None,
                   fwhm_MHz=60.0, lorentz_fraction=0.3,
                   n_rep=60, ref_time_h=24.0):
    """Build Table-2 rows: required hours per isotope at an assumed yield.

    ``rates_pps`` maps isotope -> stopped-beam rate. A single assumed
    linewidth ``fwhm_MHz`` is used for ALL isotopes so the comparison is fair
    (set it to the offline-measured ~377 MHz for a conservative estimate, or
    to the narrower line RISE is expected to deliver). Even-even isotopes are
    timed to the isotope-shift target (centroid); odd-A isotopes to the
    hyperfine-A target (the more demanding goal), which also delivers the IS.
    Returns a list of dict rows.
    """
    if isotopes is None:
        isotopes = [k for k in ISOTOPES if k != "32S"]
    rows = []
    for name in isotopes:
        spec = ISOTOPES[name]
        meta = ISOTOPE_META.get(name, {})
        rate = rates_pps.get(name, np.nan)
        even = spec["I"] == 0
        fwhm = fwhm_MHz
        obs = "centroid" if even else "A_lower"
        target = target_IS_MHz if even else target_A_MHz
        if np.isfinite(rate) and rate > 0:
            hours, sig_ref = required_hours(
                spec, rate, total_efficiency, background_cps, target,
                observable=obs, ref_time_h=ref_time_h, n_rep=n_rep,
                fwhm_MHz=fwhm, lorentz_fraction=lorentz_fraction)
        else:
            hours, sig_ref = np.nan, np.nan
        rows.append({
            "isotope": name, "Jpi": meta.get("Jpi", ""),
            "half_life": meta.get("half_life", ""),
            "rate_pps": rate, "goal": meta.get("goal", ""),
            "target_obs": obs, "target_MHz": target,
            "sigma_at_ref_MHz": sig_ref, "required_hours": hours,
        })
    return rows


def print_beamtime_table(rows):
    head = f"{'Isotope':<8}{'Jpi':<9}{'T1/2':<10}{'Rate(pps)':<11}{'Goal':<11}{'Req. hours':<11}"
    print(head); print("-" * len(head))
    total = 0.0
    for r in rows:
        h = r["required_hours"]
        if not np.isfinite(h):
            hr = "n/a"
        elif h < 0.5:
            hr = "<1"
        elif h < 10:
            hr = f"{h:.1f}"
        else:
            hr = f"{h:.0f}"
        if np.isfinite(h):
            total += h
        rate = f"{r['rate_pps']:.0f}" if np.isfinite(r["rate_pps"]) else "TBD"
        print(f"{r['isotope']:<8}{r['Jpi']:<9}{r['half_life']:<10}"
              f"{rate:<11}{r['goal']:<11}{hr:<11}")
    print("-" * len(head))
    print(f"{'Total':<49}{total:.0f} h  ({total/24:.1f} days)")


def _demo():
    # ---- Placeholder beam assumptions (mirroring the 22Si-style figure) ----
    EFF = 1.0e-3          # overall on-resonance detection efficiency (0.1%)
    BKG = 2.0e-3          # background rate (counts/s)
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))

    # Even-even: 30S, 48 h
    sim_ee = simulate_spectrum(
        ISOTOPES["30S"], stopped_beam_rate_pps=80.0, total_efficiency=EFF,
        background_cps=BKG, measurement_time_s=48 * HOUR,
        scan_range_MHz=(-400, 400), n_steps=160, fwhm_MHz=40.0, seed=1)
    _, _, info_ee = plot_simulated_spectrum(
        sim_ee, ax=axes[0], title=r"$^{30}$S  (0$^+$)")
    print(f"[30S even-even]  total={sim_ee['total_counts']:.0f} "
          f"peak={sim_ee['peak_counts']:.0f}  ->  {info_ee}")

    # Hyperfine: 29S, 48 h
    sim_hfs = simulate_spectrum(
        ISOTOPES["29S"], stopped_beam_rate_pps=120.0, total_efficiency=EFF,
        background_cps=BKG, measurement_time_s=48 * HOUR,
        scan_range_MHz=(-700, 700), n_steps=240, fwhm_MHz=40.0, seed=2)
    _, _, info_hfs = plot_simulated_spectrum(
        sim_hfs, ax=axes[1], title=r"$^{29}$S  (5/2$^+$)")
    print(f"[29S hyperfine]  total={sim_hfs['total_counts']:.0f} "
          f"peak={sim_hfs['peak_counts']:.0f}  ->  {info_hfs}")

    fig.tight_layout()
    fig.savefig("sim_spectra_examples.png", dpi=200)
    print("Saved sim_spectra_examples.png")


if __name__ == "__main__":
    _demo()
