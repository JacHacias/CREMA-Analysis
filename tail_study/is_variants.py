"""Replay the 34S-32S library rows under line-shape variants and compare the combined shift.

    python is_variants.py --variants tail_free,tail_transfer --boot 0 --jobs 4 --tag fisher
"""
import argparse, csv, json, math, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
HERE = Path(__file__).resolve().parent

VARIANTS = {
    "sym": {},
    "tail_free": {"tail_model": "exponential", "shape_transfer": "none"},
    "tail_transfer": {"tail_model": "exponential", "shape_transfer": "tail"},
    "tail_sigma_transfer": {"tail_model": "exponential", "shape_transfer": "tail+sigma"},
    "tail_transfer_flat": {"tail_model": "exponential", "shape_transfer": "tail", "fit_background_slope": False},
}


def replay(args):
    index, row, overrides = args
    import matplotlib; matplotlib.use("Agg")
    import reanalyze_library as rl_lib
    t0 = time.time()
    try:
        new = rl_lib.replay_row(row, overrides, Path(tempfile.mkdtemp()))
    except Exception as exc:
        return index, {"error": str(exc)}, time.time() - t0
    quality = json.loads(new.get("bad_scan_filter", "") or "{}").get("fit_quality", {})
    keep = {}
    for label, q in quality.items():
        if isinstance(q, dict):
            keep[label] = {k: q.get(k) for k in ("tail_fraction", "tail_fraction_unc", "tail_length_V", "tail_length_MHz",
                                                 "sigma_doppler_MHz", "sigma_doppler_V", "gamma_MHz", "deviance",
                                                 "deviance_symmetric", "fisher_center_unc_MHz", "bootstrap_center_unc_MHz",
                                                 "tail_fixed", "sigma_fixed", "reduced_chi2") if k in q}
            if "shift_disagreement_MHz" in q:
                keep[label] = dict(q)
    out = {k: new.get(k) for k in ("isotope_shift_MHz", "isotope_shift_total_unc_MHz", "isotope_shift_fit_unc_MHz",
                                  "center_reference_MHz", "center_comparison_MHz", "center_reference_total_unc_MHz",
                                  "center_comparison_total_unc_MHz")}
    out["quality"] = keep
    out["row"] = new
    return index, out, time.time() - t0


def summarize(rows_new):
    import library_uncertainty_analysis as lua
    res = lua.analyze_library_uncertainty(rows_new, lua.InclusionCuts(comparison="34S-32S"))
    f = res.get("frequentist") or {}
    return {"N": f.get("N"), "mean": f.get("weighted_mean_MHz"), "sem": f.get("weighted_scatter_sem_MHz"),
            "internal": f.get("internal_unc_MHz"), "chi2r": f.get("chi2_red"), "wstd": f.get("weighted_std_MHz"),
            "included": [f"{v['collection_date']} {v['run_label']}" for v in res.get("included", [])],
            "excluded": [f"{v['collection_date']} {v['run_label']}: {'; '.join(v.get('reasons', []))}" for v in res.get("excluded", [])],
            "bayes": res.get("bayesian")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="tail_free,tail_transfer,tail_sigma_transfer")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="fisher")
    ap.add_argument("--rows", default="")
    a = ap.parse_args()
    import reanalyze_library as rl_lib
    stored = list(csv.DictReader(open(rl_lib.LIBRARY, newline="", encoding="utf-8")))
    idx = [i for i, r in enumerate(stored) if r["comparison"] == "34S-32S"]
    if a.rows:
        idx = [int(x) for x in a.rows.split(",")]
    results = {}
    for name in a.variants.split(","):
        overrides = dict(VARIANTS[name], bootstrap_replicas=a.boot)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            outs = list(pool.map(replay, [(i, stored[i], overrides) for i in idx]))
        per_row = {}
        new_rows = []
        for i, out, dt in outs:
            per_row[i] = {k: v for k, v in out.items() if k != "row"}
            if "row" in out:
                new_rows.append(out["row"])
            s = stored[i]
            if "error" in out:
                print(f"[{name}] row {i} {s['collection_date']} {s['run_label']}: FAILED {out['error']}", flush=True)
                continue
            qs = out["quality"]
            tails = " ".join(f"{lab}: f {q.get('tail_fraction', float('nan')):.2f} lamV {q.get('tail_length_V', float('nan')):.1f} "
                             f"sdV {q.get('sigma_doppler_V', float('nan')):.2f}" for lab, q in qs.items() if "tail_fraction" in q)
            print(f"[{name}] row {i:2d} {s['collection_date']} {s['run_label'][:28]:28s} IS {out['isotope_shift_MHz']:7.1f} "
                  f"+/- {out['isotope_shift_total_unc_MHz']:5.1f} (stored {float(s['isotope_shift_MHz']):7.1f} +/- "
                  f"{float(s['isotope_shift_total_unc_MHz']):5.1f})  {tails}  [{dt:.0f}s]", flush=True)
        summ = summarize(new_rows) if new_rows else {}
        results[name] = {"overrides": overrides, "rows": per_row, "summary": summ}
        print(f"[{name}] summary: {json.dumps({k: v for k, v in summ.items() if k not in ('included', 'excluded', 'bayes')}, default=float)}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        for e in summ.get("excluded", []):
            print(f"      excluded: {e}")
        json.dump(results, open(HERE / f"is_variants_{a.tag}.json", "w"), indent=1, default=float)
        # save the replayed rows for later combination with other error models
        with open(HERE / f"is_variants_{a.tag}_{name}_rows.json", "w") as fh:
            json.dump(new_rows, fh, default=float)
