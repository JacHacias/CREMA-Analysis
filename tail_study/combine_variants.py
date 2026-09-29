"""Combine the 34S-32S pairs (same 9 pairs / 7 run groups as the adopted result) per variant."""
import csv, json, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import library_uncertainty_analysis as lua, reanalyze_library as rl_lib
HERE = Path(__file__).resolve().parent
stored = list(csv.DictReader(open(rl_lib.LIBRARY, newline="", encoding="utf-8")))
pb = {r["row"]: r for r in json.load(open(HERE / "pair_boot_100.json"))}
INCLUDED = [1, 2, 3, 6, 7, 8, 10, 12, 13]


def combine(values, sigmas, rows):
    views = [{"shift_MHz": v, "fit_unc_MHz": s, "run_label": stored[i]["run_label"], "collection_date": stored[i]["collection_date"],
              "analysis_id": stored[i]["analysis_id"]} for v, s, i in zip(values, sigmas, rows)]
    groups = lua.collapse_runs(views)
    gv = np.array([g["value_MHz"] for g in groups]); gs = np.array([g["sigma_MHz"] for g in groups])
    r = lua.weighted_mean_with_scatter_sem(gv, gs)
    bayes = lua.bayesian_random_effects_grid(gv, gs)
    return r, groups, bayes


for label, rows in (("adopted 9 pairs (05-13 excluded)", INCLUDED), ("with 05-13", sorted(INCLUDED + [11]))):
    print(f"\n=== {label} ===")
    for name, get in (("stored v2 symmetric (per-isotope boot, 200)", lambda i: (float(stored[i]["isotope_shift_MHz"]), float(stored[i]["isotope_shift_total_unc_MHz"]))),
                      ("v2 symmetric, pair bootstrap (100)", lambda i: (pb[i]["sym"]["IS"], pb[i]["sym"]["boot_sd"])),
                      ("tail transfer, pair bootstrap (100)", lambda i: (pb[i]["tail_transfer"]["IS"], pb[i]["tail_transfer"]["boot_sd"]))):
        vals = [get(i) for i in rows]
        r, groups, bayes = combine([v for v, _ in vals], [s for _, s in vals], rows)
        best = max(r["internal_unc_MHz"], r["weighted_scatter_sem_MHz"])
        print(f"  {name:44s} N_groups={r['N']}  mean {r['weighted_mean_MHz']:6.1f}  internal {r['internal_unc_MHz']:4.1f}  "
              f"scatter SEM {r['weighted_scatter_sem_MHz']:4.1f}  chi2r {r['chi2_red']:.2f}  -> {r['weighted_mean_MHz']:.1f}({best:.1f})  "
              f"Bayes {bayes.get('mu_mean', bayes.get('mean', float('nan'))):.1f}({bayes.get('mu_sd', bayes.get('sd', float('nan'))):.1f})")
        print("      groups: " + "; ".join(f"{g['collection_date']} {g['value_MHz']:.1f}({g['sigma_MHz']:.1f})" for g in groups))
