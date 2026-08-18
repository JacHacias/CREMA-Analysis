"""Dead-time propagation through the isotope-shift library.

For every library pair with raw scan CSVs on disk:
  1. Build the per-frequency-bin spectrum (in-gate hits, shots) for each
     isotope using the row's recorded ToF gates.
  2. Apply the first-order non-paralyzable dead-time correction with
     tau_d = 40 ns:  m_true = m_obs * (1 + C * m_obs), where
     C = tau_d * integral f(t)^2 dt uses the file's own measured in-bunch
     ToF envelope f (so the pairwise-overlap probability is empirical).
  3. Fit Gaussian+constant centroids to the corrected and uncorrected
     spectra identically; the centroid difference is the dead-time pull.
  4. Report the isotope-shift change dIS = dnu(heavy) - dnu(32S) per pair.

Frequencies in total (doubled) MHz to match the library's IS values.
"""
import collections
import csv
import json
import os

import numpy as np
from scipy.optimize import curve_fit

TAU_D = 40e-9
C_CM_MHZ = 29979.2458
LIB = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis\hfs_gui\data_library\isotope_shift_library.csv"


def load_scan(path):
    """Return dict bin -> [shots set, hits list of (tof, wn)] and all wn."""
    per_bin = collections.defaultdict(lambda: [set(), []])
    wns = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = row.get("scan_bin_index", "0")
            per_bin[k][0].add(row.get("bunch_id", ""))
            try:
                wn = float(row["wavemeter_wn1"])
            except (KeyError, ValueError):
                wn = None
            if wn:
                wns.append(wn)
            try:
                tof = float(row["tof"])
            except (KeyError, ValueError):
                continue
            if tof > 0.1e-6 and wn:
                per_bin[k][1].append((tof, wn))
    return per_bin, np.array(wns)


def envelope_C(tofs, gate):
    """C = tau_d * int f^2 dt from the measured in-gate ToF envelope."""
    sel = tofs[(tofs >= gate[0]) & (tofs <= gate[1])]
    if len(sel) < 200:
        return None
    h, edges = np.histogram(sel, bins=60)
    dt = edges[1] - edges[0]
    f = h / (len(sel) * dt)              # normalized density (1/s)
    return TAU_D * np.sum(f ** 2) * dt


def spectrum(per_bin, gate):
    """Arrays (freq_MHz_total, shots, hits) per scan bin."""
    F, S, H = [], [], []
    for k, (shots, hits) in per_bin.items():
        if len(shots) < 10:
            continue
        wg = [wn for t, wn in hits if gate[0] <= t <= gate[1]]
        allwn = [wn for _, wn in hits]
        if not allwn:
            continue
        F.append(np.median(allwn) * 2 * C_CM_MHZ)
        S.append(len(shots))
        H.append(len(wg))
    idx = np.argsort(F)
    return (np.array(F)[idx], np.array(S)[idx], np.array(H)[idx])


def centroid(F, counts):
    """Gaussian + constant fit; returns centroid MHz (nan on failure)."""
    if counts.sum() < 30 or len(F) < 8:
        return np.nan
    f0 = F[np.argmax(counts)]
    w = max((F.max() - F.min()) / 6, 30.0)

    def g(x, a, mu, s, c):
        return a * np.exp(-0.5 * ((x - mu) / s) ** 2) + c

    try:
        p, _ = curve_fit(
            g, F, counts,
            p0=[counts.max(), f0, w, max(counts.min(), 0.1)],
            sigma=np.sqrt(np.clip(counts, 1, None)), maxfev=20000)
        return p[1]
    except Exception:
        return np.nan


def main():
    rows = list(csv.DictReader(open(LIB, newline="", encoding="utf-8")))
    print(f"{'pair':38s} {'iso':4s} {'m_pk':>5s} {'dNu_unc->corr (MHz)':>20s}")
    summary = []
    for r in rows:
        files = r["files"].split(";")
        if not all(os.path.exists(f) for f in files):
            continue
        o = json.loads(r["options_json"])
        gates = {k: (v[0] * 1e-6, v[1] * 1e-6)
                 for k, v in o["per_isotope_tof_gates"].items()}
        windows = o["isotope_wavenumber_windows"]
        label = f"{r['collection_date']} {r['run_label'][:20]}"
        pulls = collections.defaultdict(list)
        mpk = {}
        for path in files:
            per_bin, wns = load_scan(path)
            med = np.median(wns)
            iso = next((k for k, (lo, hi) in windows.items()
                        if lo <= med <= hi), None)
            if iso is None:
                continue
            gate = gates[iso]
            F, S, H = spectrum(per_bin, gate)
            if len(F) < 8:
                pulls[iso].append(0.0)
                mpk[iso] = 0.0
                continue
            m_obs = H / S
            tofs = np.array([t for k in per_bin
                             for t, _ in per_bin[k][1]])
            C = envelope_C(tofs, gate)
            if C is None:
                C = 0.04 / TAU_D * TAU_D  # fallback ~ Gaussian 300 ns
                C = 0.0376
            m_corr = m_obs * (1 + C * m_obs)
            c_unc = centroid(F, H.astype(float))
            c_cor = centroid(F, m_corr * S)
            pull = (c_cor - c_unc) if np.isfinite(c_cor + c_unc) else np.nan
            pulls[iso].append(pull)
            mpk[iso] = m_obs.max()
        if "32S" not in pulls:
            continue
        heavy = "34S" if "34S" in pulls else ("36S" if "36S" in pulls else None)
        if heavy is None:
            continue
        d32 = np.nanmean(pulls["32S"])
        dh = np.nanmean(pulls[heavy])
        dIS = dh - d32
        unc = float(r["isotope_shift_total_unc_MHz"])
        print(f"{label:38s} 32S  {mpk.get('32S',0):5.2f} {d32:+8.3f}")
        print(f"{'':38s} {heavy:4s} {mpk.get(heavy,0):5.2f} {dh:+8.3f}"
              f"   => dIS = {dIS:+.3f} MHz  (pair unc {unc:.1f})")
        summary.append((label, dIS, unc))
    if summary:
        d = np.array([s[1] for s in summary])
        print("\npairs:", len(summary))
        print(f"dead-time IS pull: mean {np.nanmean(d):+.3f} MHz, "
              f"max |pull| {np.nanmax(np.abs(d)):.3f} MHz")
        print("vs 5.1 MHz scan-to-scan scatter and ~5-13 MHz per-pair unc")


if __name__ == "__main__":
    main()
