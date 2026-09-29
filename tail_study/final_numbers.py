"""Numbers quoted in the thesis/deck for the adopted analysis (tail model, +190.8 V), from the live library.

* library combination through the GUI's own path (cuts, run collapse, Bayes)
* 05-13 variant (manual exclusion lifted) and its pull
* line-shape model systematic: the symmetric analysis with ITS OWN v2 calibration
  (+186.5 V; stored v2 rows shifted by d(IS)/dV x (186.54 - 184.54)), combined with the
  SAME weights as the tail analysis
* charge radius via the GUI (beam-energy systematic from the energy library)
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "hfs_gui"))
import matplotlib; matplotlib.use("Agg")
import library_uncertainty_analysis as lua
import spectrum_library_gui as gui

HERE = Path(__file__).resolve().parent
live = list(csv.DictReader(open(REPO / "hfs_gui/data_library/isotope_shift_library.csv", encoding="utf-8")))
sym_rows = list(csv.DictReader(open(REPO / "hfs_gui/data_library/isotope_shift_library_before_tail_20260929_114354.csv", encoding="utf-8")))
cal = json.load(open(HERE / "recalibrate.json"))["summary"]
sysinfo = gui.beam_energy_systematic("34S-32S")
dis_dv = sysinfo["d_is_dv_MHz_per_V"]
out = {}

res = lua.analyze_library_uncertainty(live, lua.InclusionCuts(comparison="34S-32S"))
f, b = res["frequentist"], res["bayesian"]
stat = max(f["internal_unc_MHz"], f["weighted_scatter_sem_MHz"])
out["tail"] = dict(mean=f["weighted_mean_MHz"], internal=f["internal_unc_MHz"], scatter=f["weighted_scatter_sem_MHz"], stat=stat,
                   chi2r=f["chi2_red"], birge=math.sqrt(f["chi2_red"]), N=f["N"], bayes=b.get("mu_mean_MHz"), bayes_sd=b.get("mu_sd_MHz"),
                   groups=[(g["collection_date"], g["value_MHz"], g["sigma_MHz"]) for g in res["groups"]])
print(f"TAIL (adopted): {f['weighted_mean_MHz']:.2f} stat {stat:.2f} (internal {f['internal_unc_MHz']:.2f}, scatter {f['weighted_scatter_sem_MHz']:.2f}), "
      f"chi2r {f['chi2_red']:.2f}, Birge {math.sqrt(f['chi2_red']):.2f}, N={f['N']}; Bayes {b.get('mu_mean_MHz'):.2f}({b.get('mu_sd_MHz'):.2f})")
for g in res["groups"]:
    print(f"   group {g['collection_date']} {g['label'][:30]:30s} {g['value_MHz']:7.2f} +/- {g['sigma_MHz']:5.2f} ({g['n_rows']} rows)")

# 05-13 variant: lift the manual exclusion
lifted = [dict(r, notes=r["notes"].replace("[excluded]", "(was excluded)")) if r["collection_date"] == "2026-05-13" else r for r in live]
res13 = lua.analyze_library_uncertainty(lifted, lua.InclusionCuts(comparison="34S-32S"))
f13 = res13["frequentist"]
row13 = next(r for r in live if r["collection_date"] == "2026-05-13")
v13, s13 = float(row13["isotope_shift_MHz"]), float(row13["isotope_shift_total_unc_MHz"])
pull = (v13 - f["weighted_mean_MHz"]) / math.hypot(s13, f["internal_unc_MHz"])
out["with_0513"] = dict(mean=f13["weighted_mean_MHz"], stat=max(f13["internal_unc_MHz"], f13["weighted_scatter_sem_MHz"]), chi2r=f13["chi2_red"],
                        pull=pull, value=v13, sigma=s13, excluded=[e["reasons"] for e in res13["excluded"] if e["collection_date"] == "2026-05-13"])
print(f"with 05-13: {f13['weighted_mean_MHz']:.2f}({out['with_0513']['stat']:.2f}), chi2r {f13['chi2_red']:.2f}; 05-13 row {v13:.1f}({s13:.1f}) pull {pull:+.1f} sigma; "
      f"shift of mean {f13['weighted_mean_MHz'] - f['weighted_mean_MHz']:+.2f} MHz")

# symmetric comparator with its own calibration, same weights as the tail analysis
shift_sym = dis_dv * (cal["sym"]["mean"] - 184.54201214242858)
included = {e["analysis_id"] for e in res["included"]}
views_t, views_s = [], []
for rt, rs_ in zip(live, sym_rows):
    if rt["analysis_id"] not in included:
        continue
    assert rt["analysis_id"] == rs_["analysis_id"]
    base = {"run_label": rt["run_label"], "collection_date": rt["collection_date"], "analysis_id": rt["analysis_id"]}
    sig = float(rt["isotope_shift_total_unc_MHz"])
    views_t.append(dict(base, shift_MHz=float(rt["isotope_shift_MHz"]), fit_unc_MHz=sig))
    views_s.append(dict(base, shift_MHz=float(rs_["isotope_shift_MHz"]) + shift_sym, fit_unc_MHz=sig))
gt, gs_ = lua.collapse_runs(views_t), lua.collapse_runs(views_s)
wt = np.array([1 / g["sigma_MHz"] ** 2 for g in gt])
mt = float(np.sum(wt * [g["value_MHz"] for g in gt]) / wt.sum()); ms = float(np.sum(wt * [g["value_MHz"] for g in gs_]) / wt.sum())
# the symmetric analysis with its own (stored bootstrap) errors, for reference
views_s_own = [dict(v, fit_unc_MHz=float(r["isotope_shift_total_unc_MHz"])) for v, r in
               zip(views_s, [r for r in sym_rows if r["analysis_id"] in included])]
gso = lua.collapse_runs(views_s_own)
rso = lua.weighted_mean_with_scatter_sem(np.array([g["value_MHz"] for g in gso]), np.array([g["sigma_MHz"] for g in gso]))
out["model_sys"] = dict(tail_same_weights=mt, sym_same_weights=ms, diff=mt - ms, sym_own_mean=rso["weighted_mean_MHz"],
                        sym_own_stat=max(rso["internal_unc_MHz"], rso["weighted_scatter_sem_MHz"]), sym_offset_shift_MHz=shift_sym)
print(f"symmetric, own calibration (+{cal['sym']['mean']:.2f} V, +{shift_sym:.2f} MHz per row): own errors {rso['weighted_mean_MHz']:.2f}"
      f"({out['model_sys']['sym_own_stat']:.2f}); with the tail weights {ms:.2f} vs tail {mt:.2f} -> model difference {mt - ms:+.2f} MHz")

# charge radius through the GUI
cr = gui.compute_charge_radii({})
entry = next(e for e in cr["results"] if e["comparison"] == "34S-32S") if isinstance(cr, dict) and "results" in cr else None
out["gui_charge_radius"] = entry
print("GUI charge radius entry keys:", list(entry.keys()) if entry else cr)
out["beam_sys"] = sysinfo
json.dump(out, open(HERE / "final_numbers.json", "w"), indent=1, default=float)
