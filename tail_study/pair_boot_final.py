"""Pair bootstrap of the isotope shift: resample BOTH isotopes' passes in each replica.

For shape transfer the 34S fit takes the tail of the same replica's 32S fit, so the
shared tail uncertainty cancels in the difference as it should. Compared with the
symmetric model under the identical resampling. Replica fits are warm-started.
"""
import csv, json, math, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
HERE = Path(__file__).resolve().parent
B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OFFSETS = {"sym": float(sys.argv[2]) if len(sys.argv) > 2 else None, "tail_transfer": float(sys.argv[3]) if len(sys.argv) > 3 else None}
MODELS = {"sym": {}, "tail_transfer": {"tail_model": "exponential", "shape_transfer": "tail"},
          "tail_fixed_session": None}


def robust_sd(v):
    v = v[np.isfinite(v)]
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    keep = v[np.abs(v - med) <= 5 * mad] if mad > 0 else v
    return float(np.std(keep, ddof=1)), int(keep.size)


def warm_fit(rs, spec, mass, opts, nominal, index=None):
    sub = spec if index is None else spec.subset(index)
    inputs = rs._fit_inputs(sub, mass, opts, nu_ref=nominal["nu_ref_GHz"] * 1000.0, ripple_hw=nominal["fit_params"]["ripple_halfwidth"])
    fit = rs.fit_poisson_lineshape(inputs["fx"], inputs["fn"], inputs["fe"], **rs._lineshape_kwargs(inputs), start=nominal["fit_params"])
    q = {}
    if inputs["tail_model"] != "none":
        s = inputs["slope_MHz_per_V"]
        q = {"tail_fraction": fit.params["tail_fraction"], "tail_length_V": fit.params["tail_length"] / s,
             "sigma_doppler_V": fit.params["sigma_doppler"] / s}
    return nominal["nu_ref_GHz"] * 1000.0 + fit.params["center"], {"fit_quality": q}


def run_row(args):
    index, row = args
    import matplotlib; matplotlib.use("Agg")
    import quick_isotope_shift as qis, rate_spectrum as rs, reanalyze_library as rl_lib
    t0 = time.time()
    options = json.loads(row["options_json"])
    files = [rl_lib._resolve(f) for f in row["files"].split(";") if f]
    labels = rl_lib._labels(files, options)
    gates = {k: tuple(v) for k, v in options["per_isotope_tof_gates"].items()}
    prep = dict(options); prep.update(rs.V2_DEFAULTS); prep["bootstrap_replicas"] = 0
    specs = [qis._prepare_cut_file_for_label(lab, [f], options=prep, per_isotope_tof_gates=gates)[0]
             for f, lab in zip(files, labels)]
    masses = [qis.SULFUR_MASSES_U[lab] for lab in labels]
    if len(files) == 3:  # bracket: 32S before, 34S, 32S after (time-ordered)
        t = [qis._file_time_key(f) for f in files]
        w = min(max((t[1] - t[0]) / max(t[2] - t[0], 1.0), 0.0), 1.0)
        ref_idx, cmp_idx, weights = [0, 2], 1, [1 - w, w]
    else:
        ref_idx = [labels.index("32S")]; cmp_idx = 1 - ref_idx[0]; weights = [1.0]
    out = {"row": index, "date": row["collection_date"], "run": row["run_label"], "stored": float(row["isotope_shift_MHz"]),
           "stored_unc": float(row["isotope_shift_total_unc_MHz"])}
    rng_seed = (20260929 ^ zlib.crc32(row["analysis_id"].encode())) & 0x7FFFFFFF
    for model in ("sym", "tail_transfer"):
        base = dict(prep)
        base.update(MODELS[model])
        if OFFSETS.get(model) is not None:
            base["voltage_offset_V"] = OFFSETS[model]
        noms = {i: rs.fit_rate_spectrum(specs[i], masses[i], base) for i in ref_idx}
        opts_cmp = rs.transferred_shape_options(base, [noms[i] for i in ref_idx], weights)
        nom_cmp = rs.fit_rate_spectrum(specs[cmp_idx], masses[cmp_idx], opts_cmp)
        ref_c = sum(wt * noms[i]["center_abs_GHz"] * 1000.0 for wt, i in zip(weights, ref_idx))
        is0 = nom_cmp["center_abs_GHz"] * 1000.0 - ref_c
        if labels[cmp_idx] == "32S":
            is0 = -is0
        strata = {i: rs.bootstrap_strata(specs[i], 0) for i in ref_idx + [cmp_idx]}
        rng = np.random.default_rng(rng_seed)
        vals = np.full(B, np.nan)
        for b in range(B):
            try:
                idx = {i: np.concatenate([u[k] for u in strata[i] for k in rng.integers(0, len(u), len(u))]) for i in strata}
                refs = {i: warm_fit(rs, specs[i], masses[i], base, noms[i], idx[i]) for i in ref_idx}
                oc = rs.transferred_shape_options(base, [refs[i][1] for i in ref_idx], weights)
                c_cmp, _ = warm_fit(rs, specs[cmp_idx], masses[cmp_idx], oc, nom_cmp, idx[cmp_idx])
                v = c_cmp - sum(wt * refs[i][0] for wt, i in zip(weights, ref_idx))
                vals[b] = -v if labels[cmp_idx] == "32S" else v
            except Exception:
                continue
        sd, used = robust_sd(vals)
        factor = math.sqrt(np.median([len(u) / (len(u) - 1) for i in strata for u in strata[i] if len(u) > 1]))
        out[model] = {"IS": is0, "boot_sd": sd * factor, "used": used}
    out["elapsed"] = time.time() - t0
    return out


if __name__ == "__main__":
    import reanalyze_library as rl_lib
    stored = list(csv.DictReader(open(rl_lib.LIBRARY, newline="", encoding="utf-8")))
    rows = [(i, r) for i, r in enumerate(stored) if r["comparison"] == "34S-32S"]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(run_row, rows))
    for r in res:
        print(f"row {r['row']:2d} {r['date']} {r['run'][:30]:30s} stored {r['stored']:6.1f}+/-{r['stored_unc']:4.1f} | "
              f"sym {r['sym']['IS']:6.1f}+/-{r['sym']['boot_sd']:4.1f} | tail_transfer {r['tail_transfer']['IS']:6.1f}+/-{r['tail_transfer']['boot_sd']:4.1f} "
              f"(used {r['tail_transfer']['used']}) [{r['elapsed']:.0f}s]")
    json.dump(res, open(HERE / f"pair_boot_{B}{'_cal' if OFFSETS['sym'] else ''}.json", "w"), indent=1, default=float)
