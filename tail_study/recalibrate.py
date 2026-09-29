"""Collinear/anticollinear beam-energy offset with the v2 line shapes, in the rest frame.

Each calibration scan is fitted in its Doppler-corrected frame (per-bunch DMM voltage +
trial offset delta0, the scan's own geometry). The offset that makes the collinear and
anticollinear rest-frame centers agree is delta = delta0 + (c_col - c_anti)/(s_col + s_anti)
(the centers move by -/+ s delta for collinear/anticollinear). Models: symmetric v2, and v2
with the exponential energy-loss tail (core center; tail side from the geometry).
Bootstrap (pass blocks) errors on each center.
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import quick_isotope_shift as qis, rate_spectrum as rs

D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
SCANS = {"111137": ("20260617", "collinear"), "123956": ("20260617", "collinear"), "171758": ("20260617", "anticollinear"),
         "172745": ("20260617", "anticollinear"), "095024": ("20260618", "anticollinear"), "110246": ("20260618", "collinear"),
         "142931": ("20260618", "collinear"), "153348": ("20260618", "anticollinear"), "163309": ("20260618", "collinear"),
         "171531": ("20260618", "collinear")}
PAIRS = [(["111137", "123956"], ["171758", "172745"]), (["110246"], ["095024"]), (["142931"], ["153348"]),
         (["163309"], ["153348"]), (["171531"], ["153348"])]
DELTA0 = 184.54201214242858
MASS = qis.SULFUR_MASSES_U["32S"]
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 100

res = {}
for stamp, (day, geom) in SCANS.items():
    path = Path(D) / f"scan_{day}_{stamp}.csv"
    base = dict(qis.DEFAULT_ANALYSIS_OPTIONS)
    base.update(voltage_offset_V=DELTA0, validate_isotope_wavenumber=False, geometry=geom, bootstrap_replicas=BOOT)
    spec, _ = qis._prepare_cut_file_for_label("32S", [path], options=base, per_isotope_tof_gates={"32S": (4.25, 5.5)})
    res[stamp] = {"geom": geom}
    for model, extra in (("sym", {}), ("tail", {"tail_model": "exponential"})):
        o = dict(base, **extra)
        r = rs.apply_bootstrap(rs.fit_rate_spectrum(spec, MASS, o), spec, MASS, o)
        q = r["fit_quality"]
        inp = rs._fit_inputs(spec, MASS, o, nu_ref=r["nu_ref_GHz"] * 1000.0)
        res[stamp][model] = dict(c=r["center_abs_GHz"] * 1000.0, e=r["center_fit_unc_GHz"] * 1000.0,
                                 fisher=q.get("fisher_center_unc_MHz"), s=inp["slope_MHz_per_V"], side=inp["tail_side"],
                                 f=q.get("tail_fraction"), lamV=q.get("tail_length_V"), dev=q.get("deviance"),
                                 dev_sym=q.get("deviance_symmetric"), chi2=q["reduced_chi2"])
    t = res[stamp]["tail"]
    print(f"{day}_{stamp} {geom:13s} n={spec.num_points:6d} sym c {res[stamp]['sym']['c'] - 756363000:+8.1f} +/- {res[stamp]['sym']['e']:4.1f} "
          f"(chi2 {res[stamp]['sym']['chi2']:.1f}) | tail c {t['c'] - 756363000:+8.1f} +/- {t['e']:4.1f} (chi2 {t['chi2']:.1f}, side {t['side']:+d}, "
          f"f {t['f']:.2f}, lam {t['lamV']:.1f} V, dDev {t['dev_sym'] - t['dev']:.0f})", flush=True)

print(f"\nOffset delta (V) per pair [rest frame, per-bunch HV], stored library value {DELTA0:.2f}:")
summary = {}
for model in ("sym", "tail"):
    deltas, errs = [], []
    for cols, antis in PAIRS:
        cc = np.mean([res[s][model]["c"] for s in cols]); ec = math.sqrt(sum(res[s][model]["e"] ** 2 for s in cols)) / len(cols)
        ca = np.mean([res[s][model]["c"] for s in antis]); ea = math.sqrt(sum(res[s][model]["e"] ** 2 for s in antis)) / len(antis)
        sc = np.mean([res[s][model]["s"] for s in cols]); sa = np.mean([res[s][model]["s"] for s in antis])
        deltas.append(DELTA0 + (cc - ca) / (sc + sa)); errs.append(math.hypot(ec, ea) / (sc + sa))
    deltas, errs = np.array(deltas), np.array(errs)
    # pairs 3-5 share one anticollinear scan: collapse them before the mean (like the GUI's clusters)
    clusters = [deltas[0], deltas[1], np.mean(deltas[2:])]
    cl_err = [errs[0], errs[1], np.mean(errs[2:])]
    w = 1 / np.array(cl_err) ** 2
    wm = float(np.sum(w * clusters) / np.sum(w)); sem = float(np.std(clusters, ddof=1) / math.sqrt(len(clusters)))
    summary[model] = dict(pairs=deltas.tolist(), errs=errs.tolist(), clusters=clusters, mean=float(np.mean(clusters)), wmean=wm, sem=sem)
    print(f"  {model:5s} pairs " + "  ".join(f"{d:7.2f}({e:.2f})" for d, e in zip(deltas, errs)) +
          f"  | 3 clusters mean {np.mean(clusters):.2f}, weighted {wm:.2f}, scatter SEM {sem:.2f} V")
print(f"\n  tail - sym offset: {summary['tail']['mean'] - summary['sym']['mean']:+.2f} V (cluster means)")
json.dump({"scans": res, "summary": summary}, open("recalibrate.json", "w"), indent=1, default=float)
