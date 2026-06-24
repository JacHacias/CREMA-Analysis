"""
Diagnostic (NOT part of the production pipeline): compare a symmetric Voigt against
an asymmetric split-Voigt (separate Gaussian width left/right of the peak) for the
32/34S resonance centroids, across every 34S-32S library pair.

For each scan we refit the centroid both ways using the pipeline's own frequency
calibration and the library's recorded per-isotope ToF gate, then compute the isotope
shift IS = center(34S) - center(32S) both ways and compare the run-to-run scatter.
The question is not just "does the asymmetric model fit better" (it usually does) but
"does it give a more reproducible IS, and is its centroid shift consistent between
isotopes so it cancels in the difference?"

Usage:  python lineshape_model_comparison.py
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

import isotope_shift_analysis as two_fit
from quick_isotope_shift import (
    concatenate_cut_files,
    remove_bad_scans,
    _rest_frequencies_for_label,
    DEFAULT_ANALYSIS_OPTIONS,
)

DATA_DIR = Path(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data")
LIBRARY_CSV = Path("hfs_gui/data_library/isotope_shift_library.csv")
WINDOW_MHZ = 1000.0


def assign_isotope(median_wn: float, windows: dict) -> str | None:
    iso, best = None, 1e9
    for cand, (lo, hi) in windows.items():
        if lo <= median_wn <= hi:
            return cand
        d = min(abs(median_wn - lo), abs(median_wn - hi))
        if d < best:
            best, iso = d, cand
    return iso


def _voigt_profile(x, ctr, sg, gl):
    z = ((x - ctr) + 1j * gl) / (sg * np.sqrt(2.0))
    return np.real(wofz(z)) / (sg * np.sqrt(2.0 * np.pi))


def split_voigt(x, amp, ctr, sg_l, sg_r, gl, bg):
    out = np.empty_like(x, dtype=float)
    left = x < ctr
    pk_l = _voigt_profile(np.array([0.0]), 0.0, sg_l, gl)[0]
    pk_r = _voigt_profile(np.array([0.0]), 0.0, sg_r, gl)[0]
    out[left] = amp * _voigt_profile(x[left], ctr, sg_l, gl) / pk_l
    out[~left] = amp * _voigt_profile(x[~left], ctr, sg_r, gl) / pk_r
    return bg + out


def fit_scan(path: Path, iso: str, options: dict, gate) -> dict:
    dat = concatenate_cut_files([path])
    if options.get("auto_remove_bad_scans", True):
        dat, _ = remove_bad_scans(
            dat, scan_bin_col="scan_bin_index", wn_col=options.get("wn_col", "wavemeter_wn1"),
            min_coverage_fraction=options.get("bad_scan_min_coverage_fraction", 0.60),
            min_points_fraction=options.get("bad_scan_min_points_fraction", 0.35),
            max_spectrum_peak_z=options.get("bad_scan_max_spectrum_peak_z", 6.0),
        )
    dat = two_fit.apply_tof_gate(dat, tof_gate_us=gate, tof_col=options.get("tof_col", "tof"))
    x_abs = _rest_frequencies_for_label(dat, iso, options)  # GHz
    nu_ref = float(np.median(x_abs))
    x = (x_abs - nu_ref) * 1000.0  # MHz
    sel = np.abs(x) < WINDOW_MHZ
    counts, edges = np.histogram(x[sel], bins=int(2 * WINDOW_MHZ / 20.0), range=(-WINDOW_MHZ, WINDOW_MHZ))
    c = 0.5 * (edges[:-1] + edges[1:])
    yerr = np.sqrt(np.clip(counts, 1.0, None))
    ref_MHz = nu_ref * 1000.0

    ps, _pc, pe, xf = two_fit.fit_histogram_peak(c, counts)
    ms = two_fit.voigt(c, *ps)
    chi_s = float(np.sum(((counts - ms) / yerr) ** 2) / max(len(c) - 5, 1))

    out = {"n": int(sel.sum()), "sym_center": ref_MHz + ps[1], "sym_chi2": chi_s,
           "split_center": np.nan, "split_chi2": np.nan, "wL": np.nan, "wR": np.nan}
    try:
        p0 = [ps[0], ps[1], abs(ps[2]) or 100.0, abs(ps[2]) or 100.0, abs(ps[3]) or 50.0, ps[4]]
        pp, _ = curve_fit(split_voigt, c, counts, p0=p0, sigma=yerr, absolute_sigma=True, maxfev=20000)
        msp = split_voigt(c, *pp)
        out["split_center"] = ref_MHz + pp[1]
        out["split_chi2"] = float(np.sum(((counts - msp) / yerr) ** 2) / max(len(c) - 6, 1))
        out["wL"], out["wR"] = abs(pp[2]), abs(pp[3])
    except Exception:
        pass
    return out


def main() -> int:
    rows = list(csv.DictReader(open(LIBRARY_CSV, encoding="utf-8")))
    is_sym, is_split = [], []
    print(f"{'date':<11}{'run':<22}{'IS sym':>9}{'IS split':>10}{'shift':>8}   per-scan (iso: dCenter MHz, chi2 sym->split, wL/wR)")
    for r in rows:
        if r.get("comparison") != "34S-32S":
            continue
        opts = json.loads(r.get("options_json") or "{}")
        windows = opts.get("isotope_wavenumber_windows", {})
        gates = {k: v for k, v in (opts.get("per_isotope_tof_gates") or {}).items() if v}
        files = re.findall(r"scan_\d{8}_\d{6}\.csv", r.get("files", ""))
        if not files or not windows:
            continue
        sym = {"32S": [], "34S": []}
        spl = {"32S": [], "34S": []}
        notes = []
        ok = True
        for fn in files:
            p = DATA_DIR / fn
            if not p.exists():
                ok = False
                break
            a = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)
            iso = assign_isotope(float(np.median(np.asarray(a["wavemeter_wn1"], float))), windows)
            if iso not in ("32S", "34S") or iso not in gates:
                continue
            f = fit_scan(p, iso, opts, gates[iso])
            sym[iso].append(f["sym_center"])
            if np.isfinite(f["split_center"]):
                spl[iso].append(f["split_center"])
            ratio = (f["wL"] / f["wR"]) if (f["wR"] and np.isfinite(f["wR"])) else float("nan")
            notes.append(f"{iso}:dC{f['split_center']-f['sym_center']:+.0f} chi2 {f['sym_chi2']:.0f}->{f['split_chi2']:.0f} wL/wR={ratio:.2f}")
        if not ok or not sym["32S"] or not sym["34S"]:
            continue
        iss = (np.mean(sym["34S"]) - np.mean(sym["32S"]))
        is_sym.append(iss)
        if spl["32S"] and spl["34S"]:
            isp = (np.mean(spl["34S"]) - np.mean(spl["32S"]))
            is_split.append(isp)
        else:
            isp = float("nan")
        print(f"{r.get('collection_date',''):<11}{(r.get('run_label','') or '')[:21]:<22}{iss:>9.1f}{isp:>10.1f}{isp-iss:>8.1f}   " + " | ".join(notes))

    s = np.array(is_sym)
    p = np.array(is_split)
    print(f"\n{len(s)} pairs (symmetric), {len(p)} pairs (split).")
    if len(s) >= 2 and len(p) >= 2:
        print(f"{'':<16}{'symmetric':>12}{'split-Voigt':>12}")
        print(f"{'mean IS (MHz)':<16}{s.mean():>12.1f}{p.mean():>12.1f}")
        print(f"{'std (MHz)':<16}{s.std(ddof=1):>12.2f}{p.std(ddof=1):>12.2f}")
        print(f"\nscatter change: {(p.std(ddof=1)/s.std(ddof=1)-1)*100:+.1f}%  (negative = split-Voigt tightens IS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
