"""Final combination: symmetric vs tail-transfer, each with its own v2 beam-energy calibration."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import library_uncertainty_analysis as lua, reanalyze_library as rl_lib
HERE = Path(__file__).resolve().parent
stored = list(csv.DictReader(open(rl_lib.LIBRARY, newline="", encoding="utf-8")))
pb = {r["row"]: r for r in json.load(open(HERE / "pair_boot_200_cal.json"))}
cal = json.load(open(HERE / "recalibrate.json"))["summary"]
DIS_DV = 0.92            # MHz per V (d IS / d V, GUI beam_energy_systematic)
MS, F = 602.52, -47.91   # MHz, MHz/fm^2 (thesis Ch6)
INCLUDED = [1, 2, 3, 6, 7, 8, 10, 12, 13]


def combine(rows, get):
    views = [{"shift_MHz": get(i)[0], "fit_unc_MHz": get(i)[1], "run_label": stored[i]["run_label"],
              "collection_date": stored[i]["collection_date"], "analysis_id": stored[i]["analysis_id"]} for i in rows]
    groups = lua.collapse_runs(views)
    gv = np.array([g["value_MHz"] for g in groups]); gs = np.array([g["sigma_MHz"] for g in groups])
    r = lua.weighted_mean_with_scatter_sem(gv, gs)
    b = lua.bayesian_random_effects_grid(gv, gs)
    return r, groups, b


out = {}
for key, label, cal_key in (("sym", "symmetric v2", "sym"), ("tail", "tail + transfer", "tail")):
    get = (lambda i, m=("sym" if key == "sym" else "tail_transfer"): (pb[i][m]["IS"], pb[i][m]["boot_sd"]))
    r, groups, b = combine(INCLUDED, get)
    r13, _, _ = combine(sorted(INCLUDED + [11]), get)
    unc = max(r["internal_unc_MHz"], r["weighted_scatter_sem_MHz"])
    sys_beam = DIS_DV * cal[cal_key]["sem"]
    out[key] = dict(label=label, mean=r["weighted_mean_MHz"], unc=unc, internal=r["internal_unc_MHz"], scatter=r["weighted_scatter_sem_MHz"],
                    chi2r=r["chi2_red"], groups=groups, bayes=b, offset_V=cal[cal_key]["mean"], offset_sem_V=cal[cal_key]["sem"],
                    beam_sys=sys_beam, with_0513=dict(mean=r13["weighted_mean_MHz"], unc=max(r13["internal_unc_MHz"], r13["weighted_scatter_sem_MHz"]),
                                                       chi2r=r13["chi2_red"]))
    dr2 = (r["weighted_mean_MHz"] - MS) / F
    out[key]["dr2"] = dr2; out[key]["dr2_stat"] = unc / abs(F); out[key]["dr2_sys"] = sys_beam / abs(F)
    print(f"{label:18s} offset {cal[cal_key]['mean']:.2f}({cal[cal_key]['sem']:.2f}) V  IS = {r['weighted_mean_MHz']:.1f}({unc:.1f})stat({sys_beam:.1f})beam MHz  "
          f"[internal {r['internal_unc_MHz']:.1f}, scatter {r['weighted_scatter_sem_MHz']:.1f}, chi2r {r['chi2_red']:.2f}]  "
          f"Bayes {b.get('posterior_mean_MHz', b.get('mu_mean', float('nan')))}  with 05-13: {r13['weighted_mean_MHz']:.1f}({out[key]['with_0513']['unc']:.1f})  "
          f"dr2 = {dr2:.3f}({unc / abs(F):.3f})({sys_beam / abs(F):.3f}) fm2")
print(f"\nmodel difference tail - sym: {out['tail']['mean'] - out['sym']['mean']:+.1f} MHz")
print("bayes keys:", list(out["sym"]["bayes"].keys())[:10])
json.dump(out, open(HERE / "is_final_summary.json", "w"), indent=1, default=float)
