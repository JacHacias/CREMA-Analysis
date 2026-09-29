"""Pass-by-pass centers: symmetric model vs tail model (tail/shape fixed from the full scan).

If the ~10 MHz pass-to-pass jitter of the symmetric center is line-shape mismatch
(the center of a wrong model depends on which parts of the line each pass weights),
the tail model should shrink it; if it is real motion (HV, laser), it should not.
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import quick_isotope_shift as qis, rate_spectrum as rs

A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"; D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
cases = [("32S", D + r"\scan_20260508_131731.csv", (4.25, 5.5)), ("32S", D + r"\scan_20260508_140738.csv", (4.25, 5.5)),
         ("32S", D + r"\scan_20260508_153504.csv", (4.25, 5.5)), ("32S", D + r"\scan_20260511_083642.csv", (4.25, 5.25)),
         ("32S", D + r"\scan_20260512_144505.csv", (4.25, 5.25)), ("32S", D + r"\scan_20260513_102135.csv", (4.25, 5.25)),
         ("32S", A + r"\32S_3-23-26.csv", (4.25, 5.5)), ("32S", A + r"\32S_3-27-26.csv", (4.25, 5.5)),
         ("32S", D + r"\scan_20260601_144018.csv", (4.25, 5.5)),
         ("34S", D + r"\scan_20260512_155458.csv", (4.7, 5.4)), ("34S", D + r"\scan_20260508_134603.csv", (5.1, 5.8)),
         ("34S", A + r"\34S_3-23-26.csv", (5.1, 5.8))]
REF32 = {"scan_20260512_155458.csv": D + r"\scan_20260512_144505.csv", "scan_20260508_134603.csv": D + r"\scan_20260508_131731.csv",
         "34S_3-23-26.csv": A + r"\32S_3-23-26.csv"}


def chi2r(vals, errs):
    ok = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
    v, e = vals[ok], errs[ok]
    if v.size < 2:
        return float("nan"), float("nan")
    w = 1 / e ** 2; m = np.sum(w * v) / np.sum(w)
    return float(np.sum(((v - m) / e) ** 2) / (v.size - 1)), float(np.std(v, ddof=1))


out = {}
base = dict(qis.DEFAULT_ANALYSIS_OPTIONS); base.update(voltage_offset_V=184.54201214242858, bootstrap_replicas=0)
for lab, path, gate in cases:
    mass = qis.SULFUR_MASSES_U[lab]
    spec, _ = qis._prepare_cut_file_for_label(lab, [Path(path)], options=base, per_isotope_tof_gates={lab: gate})
    sym_full = rs.fit_rate_spectrum(spec, mass, base)
    tail_opts = dict(base, tail_model="exponential")
    if lab == "34S":  # tail from the reference 32S scan (shape transfer, as in the IS analysis)
        ref_spec, _ = qis._prepare_cut_file_for_label("32S", [Path(REF32[Path(path).name])], options=base,
                                                      per_isotope_tof_gates={"32S": (4.25, 5.25) if "0512" in path else (4.25, 5.5)})
        ref = rs.fit_rate_spectrum(ref_spec, qis.SULFUR_MASSES_U["32S"], dict(tail_opts, shape_transfer="tail"))
        tail_opts = rs.transferred_shape_options(dict(tail_opts, shape_transfer="tail"), [ref])
    tail_full = rs.fit_rate_spectrum(spec, mass, tail_opts)
    q = tail_full["fit_quality"]
    fixed_tail = dict(base, tail_model="exponential", tail_fraction=q["tail_fraction"], tail_length_V=q["tail_length_V"])
    fixed_shape = dict(fixed_tail, sigma_doppler_V=q["sigma_doppler_V"], lorentzian_hwhm_MHz=tail_full["fit_params"]["gamma"])
    fixed_sym = dict(base, lorentzian_hwhm_MHz=sym_full["fit_params"]["gamma"])
    rows = {k: ([], []) for k in ("sym", "sym_fixedgamma", "tail_fixed", "shape_fixed")}
    for p in np.unique(spec.group_pass):
        sub = spec.subset(np.flatnonzero(spec.group_pass == p))
        for key, o in (("sym", base), ("sym_fixedgamma", fixed_sym), ("tail_fixed", fixed_tail), ("shape_fixed", fixed_shape)):
            try:
                r = rs.fit_rate_spectrum(sub, mass, o)
                rows[key][0].append(r["center_abs_GHz"] * 1e3); rows[key][1].append(r["center_fit_unc_GHz"] * 1e3)
            except Exception:
                rows[key][0].append(np.nan); rows[key][1].append(np.nan)
    name = Path(path).name
    out[name] = {}
    line = f"{lab} {name:26s} passes {len(rows['sym'][0]):2d} | full Fisher: sym {sym_full['center_fit_unc_GHz'] * 1e3:4.1f}, tail {tail_full['center_fit_unc_GHz'] * 1e3:4.1f} |"
    for key, (v, e) in rows.items():
        v, e = np.array(v), np.array(e)
        c2, sd = chi2r(v, e)
        n_ok = np.isfinite(v).sum()
        scan_unc = sd / math.sqrt(max(n_ok, 1))
        out[name][key] = dict(centers=v.tolist(), errors=e.tolist(), chi2r=c2, sd=sd, scan_unc=scan_unc)
        line += f" {key}: sd {sd:5.1f} chi2r {c2:5.1f} -> {scan_unc:4.1f} |"
    print(line, flush=True)
json.dump(out, open("per_pass_tail.json", "w"), indent=1, default=float)
