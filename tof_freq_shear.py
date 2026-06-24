"""
Prototype: 2-D detuning-vs-ToF visualization and a ToF-dependent frequency
("shear") correction for sulfur RIS resonances.

Premise: collisional energy-loss straggling in the charge-exchange cell (CEC)
couples an ion's arrival time (ToF) to its resonant laser frequency. In the 2-D
(detuning, ToF) plane the resonance should be a vertical stripe (frequency
independent of ToF) but instead tilts onto a diagonal. Projecting that tilted
distribution onto the frequency axis is what makes the 1-D lineshape asymmetric
and broad. We estimate the tilt from the data and rotate it out per ion:

    detuning_corrected = detuning - slope * (tof - tof_ref)

then compare the 1-D lineshape (symmetric Voigt fit) before and after.

This is a standalone investigation; it does not modify the analysis pipeline.

Usage:
    python tof_freq_shear.py [scan_file.csv ...]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isotope_shift_analysis as isa

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data")
OUT_DIR = Path(__file__).resolve().parent / "tof_freq_analysis"
LIBRARY_CSV = Path(__file__).resolve().parent / "hfs_gui" / "data_library" / "isotope_shift_library.csv"
CM_TO_MHZ = 2.99792458e4          # 1 cm^-1 in MHz
DOUBLING = 2.0                    # frequency_multiplier in the pipeline

DEFAULT_FILES = ["scan_20260601_120201.csv"]


def library_gate_for_scan(filename: str) -> tuple[str | None, tuple[float, float] | None]:
    """Return the (isotope, ToF gate in us) the library recorded for this scan file.

    Each scan file is a single isotope; we identify which by matching its median
    wavenumber to the row's isotope_wavenumber_windows, then read that isotope's
    per_isotope_tof_gates. Returns (None, None) if the file is not in the library.
    """
    import csv as _csv
    import json as _json
    name = Path(filename).name
    if not LIBRARY_CSV.exists():
        return None, None
    for row in _csv.DictReader(open(LIBRARY_CSV, encoding="utf-8")):
        if name not in str(row.get("files", "")):
            continue
        opts = _json.loads(row.get("options_json") or "{}")
        gates = opts.get("per_isotope_tof_gates") or {}
        windows = opts.get("isotope_wavenumber_windows") or {}
        path = DATA_DIR / name
        if not path.exists() or not windows:
            return None, None
        a = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        wn = np.asarray(a["wavemeter_wn1"], dtype=float)
        mwn = float(np.median(wn[np.isfinite(wn)]))
        iso, best_d = None, 1e9
        for cand, (lo, hi) in windows.items():
            if lo <= mwn <= hi:
                iso = cand
                break
            d = min(abs(mwn - lo), abs(mwn - hi))
            if d < best_d:
                best_d, iso = d, cand
        gate = gates.get(iso)
        if gate:
            return iso, (float(gate[0]), float(gate[1]))
        return iso, None
    return None, None


def load_events(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-ion (wavenumber cm^-1, tof in microseconds) across the given files."""
    wn_all, tof_all = [], []
    for p in paths:
        a = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)
        if a.dtype.names is None or "wavemeter_wn1" not in a.dtype.names:
            continue
        wn = np.asarray(a["wavemeter_wn1"], dtype=float)
        tof = np.asarray(a["tof"], dtype=float) * 1e6
        wn_all.append(wn)
        tof_all.append(tof)
    wn = np.concatenate(wn_all)
    tof = np.concatenate(tof_all)
    # Empty DAQ rows are recorded with tof == 0; keep only real ion events.
    m = np.isfinite(wn) & np.isfinite(tof) & (tof > 0.05)
    return wn[m], tof[m]


def find_bunch_window(tof: np.ndarray, floor_frac: float = 0.08) -> tuple[float, float]:
    """Locate the dominant ion bunch in ToF and return a window that spans it plus its
    straggling tail (expanding from the peak while counts stay above floor_frac*peak)."""
    hi = np.percentile(tof, 99.8)
    counts, edges = np.histogram(tof, bins=240, range=(tof.min(), hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ipk = int(np.argmax(counts))
    thresh = floor_frac * counts[ipk]
    lo_i = ipk
    while lo_i > 0 and counts[lo_i - 1] > thresh:
        lo_i -= 1
    hi_i = ipk
    while hi_i < len(counts) - 1 and counts[hi_i + 1] > thresh:
        hi_i += 1
    # pad slightly so the tail beyond the floor is still visible in the 2-D map
    width = centers[hi_i] - centers[lo_i]
    return centers[lo_i] - 0.25 * width, centers[hi_i] + 0.5 * width


def detuning_mhz(wn: np.ndarray, ref_cm: float) -> np.ndarray:
    return (wn - ref_cm) * DOUBLING * CM_TO_MHZ


def resonance_ridge(det: np.ndarray, tof: np.ndarray, half_window_MHz: float,
                    n_tof_bins: int = 12):
    """For each ToF slice, estimate the background-subtracted resonance centroid in
    detuning. Returns (tof_centers, det_centroids, weights) for slices with signal."""
    tlo, thi = np.percentile(tof, [2, 98])
    edges = np.linspace(tlo, thi, n_tof_bins + 1)
    tof_c, det_c, wts = [], [], []
    for i in range(n_tof_bins):
        sel = (tof >= edges[i]) & (tof < edges[i + 1])
        d = det[sel]
        if d.size < 200:
            continue
        # 1-D detuning histogram in this ToF slice; subtract a flat background taken
        # from the wings, then take the weighted centroid of the central peak.
        hcounts, hedges = np.histogram(d, bins=60, range=(-4 * half_window_MHz, 4 * half_window_MHz))
        hc = 0.5 * (hedges[:-1] + hedges[1:])
        wings = np.abs(hc) > 2.0 * half_window_MHz
        bkg = np.median(hcounts[wings]) if wings.any() else 0.0
        sig = np.clip(hcounts - bkg, 0, None)
        core = np.abs(hc) <= half_window_MHz
        if sig[core].sum() < 50:
            continue
        centroid = np.sum(hc[core] * sig[core]) / np.sum(sig[core])
        tof_c.append(0.5 * (edges[i] + edges[i + 1]))
        det_c.append(centroid)
        wts.append(np.sqrt(sig[core].sum()))
    return np.array(tof_c), np.array(det_c), np.array(wts)


def voigt_fit(det: np.ndarray, window_MHz: float, bin_MHz: float = 8.0):
    """Histogram the detuning and fit a symmetric Voigt; return fit metrics."""
    sel = np.abs(det) < window_MHz
    d = det[sel]
    nb = max(20, int(2 * window_MHz / bin_MHz))
    counts, edges = np.histogram(d, bins=nb, range=(-window_MHz, window_MHz))
    x = 0.5 * (edges[:-1] + edges[1:])
    popt, _pcov, perr, x_fit = isa.fit_histogram_peak(x, counts)
    model = isa.voigt(x, *popt)
    resid = counts - model
    yerr = np.sqrt(np.clip(counts, 1.0, None))
    dof = max(len(counts) - len(popt), 1)
    chi2_red = float(np.sum((resid / yerr) ** 2) / dof)
    fwhm = isa._voigt_fwhm(popt[2], popt[3]) if hasattr(isa, "_voigt_fwhm") else 2.355 * popt[2]
    # data-driven skewness of the background-subtracted core (asymmetry proxy)
    bkg = popt[4]
    sig = np.clip(counts - bkg, 0, None)
    if sig.sum() > 0:
        mu = np.sum(x * sig) / np.sum(sig)
        var = np.sum(sig * (x - mu) ** 2) / np.sum(sig)
        skew = float(np.sum(sig * (x - mu) ** 3) / np.sum(sig) / (var ** 1.5 + 1e-12))
    else:
        skew = float("nan")
    return {
        "x": x, "counts": counts, "model": model, "x_fit": x_fit, "popt": popt,
        "center_MHz": float(popt[1]), "center_unc_MHz": float(perr[1]),
        "fwhm_MHz": float(fwhm), "chi2_red": chi2_red, "skew": skew,
    }


def analyze(paths: list[Path], tag: str) -> dict:
    wn, tof = load_events(paths)
    # Use the per-isotope ToF gate the library recorded for this scan; fall back to the
    # auto bunch-finder only if the file is not in the library.
    isotope, gate = library_gate_for_scan(paths[0].name)
    if gate is not None:
        tlo, thi = gate
        gate_source = f"library gate {isotope} [{tlo}, {thi}] us"
    else:
        tlo, thi = find_bunch_window(tof)
        gate_source = f"auto bunch window [{tlo:.2f}, {thi:.2f}] us (not in library)"
    bunch = (tof >= tlo) & (tof <= thi)
    wn, tof = wn[bunch], tof[bunch]

    # rough resonance center for the detuning reference: peak of the 1-D wn spectrum
    hcounts, hedges = np.histogram(wn, bins=120)
    ref_cm = 0.5 * (hedges[:-1] + hedges[1:])[np.argmax(hcounts)]
    det = detuning_mhz(wn, ref_cm)

    # coarse linewidth (clamped so a poor fit cannot blow up the view), then a zoomed
    # analysis window of ~3 x FWHM around the resonance.
    pre = voigt_fit(det, window_MHz=600.0, bin_MHz=20.0)
    fwhm0 = float(np.clip(pre["fwhm_MHz"], 80.0, 800.0))
    half = 0.5 * fwhm0
    win = float(np.clip(3.0 * fwhm0, 300.0, 1500.0))

    # Ridge points (for visualization) plus a robust covariance-based tilt estimate on
    # the resonance CORE only, so a spectral shoulder cannot bias the slope. The core
    # tilt is the linear regression of detuning on ToF: slope = cov(det,tof)/var(tof).
    tof_c, det_c, wts = resonance_ridge(det, tof, half_window_MHz=half)
    core = np.abs(det) < half
    if core.sum() > 200 and np.var(tof[core]) > 0:
        slope = float(np.cov(det[core], tof[core])[0, 1] / np.var(tof[core]))
    elif tof_c.size >= 3:
        slope = float(np.polyfit(tof_c, det_c, 1, w=wts)[0])
    else:
        slope = 0.0
    tof_ref = float(np.median(tof[core])) if core.any() else float(np.median(tof))
    intercept = float(np.median(det[core] - slope * (tof[core] - tof_ref))) if core.any() else 0.0
    intercept += slope * tof_ref  # express line as det = intercept + slope*tof for the overlay
    det_corr = det - slope * (tof - tof_ref)

    before = voigt_fit(det, window_MHz=win, bin_MHz=20.0)
    after = voigt_fit(det_corr, window_MHz=win, bin_MHz=20.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    extent_w = win
    for ax, dd, ttl in ((axes[0], det, "raw"), (axes[1], det_corr, "shear-corrected")):
        ax.hist2d(dd, tof, bins=[70, 50], range=[[-extent_w, extent_w], [tlo, thi]], cmap="viridis")
        ax.set_xlim(-extent_w, extent_w)
        ax.set_xlabel("detuning (MHz)"); ax.set_ylabel("ToF (us)")
        ax.set_title(f"{ttl}", fontweight="bold")
    # overlay fitted tilt on the raw panel
    tt = np.linspace(tlo, thi, 50)
    axes[0].plot(intercept + slope * tt, tt, "r-", lw=2,
                 label=f"tilt {slope:+.2f} MHz/us")
    axes[0].scatter(det_c, tof_c, c="white", s=18, edgecolors="r", zorder=5)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].axvline(0, color="w", ls=":", lw=1)
    axes[1].axvline(0, color="w", ls=":", lw=1)

    ax = axes[2]
    ax.step(before["x"], before["counts"], where="mid", color="#888", label="raw")
    ax.step(after["x"], after["counts"], where="mid", color="#0b7285", label="corrected")
    if after["x_fit"] is not None:
        ax.plot(after["x"], after["model"], "r-", lw=1.5, label="Voigt (corrected)")
    ax.set_xlabel("detuning (MHz)"); ax.set_ylabel("counts")
    ax.set_xlim(-win, win)
    ax.set_title("1-D projection", fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle(f"{tag}: tilt={slope:+.2f} MHz/us", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"{tag}_shear.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return {
        "tag": tag, "n_events": int(wn.size), "tof_window_us": (round(tlo, 2), round(thi, 2)),
        "isotope": isotope, "gate_source": gate_source,
        "slope_MHz_per_us": float(slope),
        "before": before, "after": after, "plot": str(out),
    }


def main(argv: list[str]) -> int:
    names = argv[1:] or DEFAULT_FILES
    paths = [DATA_DIR / n if not Path(n).is_absolute() else Path(n) for n in names]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No input files found.")
        return 1
    tag = Path(paths[0]).stem
    r = analyze(paths, tag)
    b, a = r["before"], r["after"]
    print(f"== {r['tag']} ==  {r['n_events']} events  [{r['gate_source']}]")
    print(f"fitted tilt: {r['slope_MHz_per_us']:+.3f} MHz/us")
    print(f"{'metric':<22}{'raw':>12}{'corrected':>12}{'change':>10}")
    for key, label, unit in (("fwhm_MHz", "Voigt FWHM", "MHz"),
                             ("chi2_red", "reduced chi^2", ""),
                             ("center_unc_MHz", "centroid unc", "MHz"),
                             ("skew", "lineshape skew", "")):
        bv, av = b[key], a[key]
        chg = (av - bv) / bv * 100 if bv else float("nan")
        print(f"{label+' '+unit:<22}{bv:>12.3f}{av:>12.3f}{chg:>9.1f}%")
    print(f"plot: {r['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
