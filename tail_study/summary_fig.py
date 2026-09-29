"""Summary figure of the width / energy-loss-tail study (2026-09-29)."""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                            "axes.unicode_minus": False, "axes.formatter.use_mathtext": True, "font.size": 10})
import quick_isotope_shift as qis, rate_spectrum as rs

HERE = Path(__file__).resolve().parent
D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 2.6], hspace=0.38, wspace=0.28)

# (a) example spectrum, symmetric vs tail
ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[1, 0], sharex=ax)
opts = dict(qis.DEFAULT_ANALYSIS_OPTIONS); opts.update(voltage_offset_V=184.542, bootstrap_replicas=0)
spec, _ = qis._prepare_cut_file_for_label("32S", [Path(D) / "scan_20260508_140738.csv"], options=opts, per_isotope_tof_gates={"32S": (4.25, 5.5)})
m = qis.SULFUR_MASSES_U["32S"]
r_sym = rs.fit_rate_spectrum(spec, m, opts); r_tail = rs.fit_rate_spectrum(spec, m, dict(opts, tail_model="exponential"))
inp = rs._fit_inputs(spec, m, opts, nu_ref=r_sym["nu_ref_GHz"] * 1000)
fx, fn, fe = inp["fx"], inp["fn"], inp["fe"]
cen, nb, eb, idx = rs._binned(fx, fn, fe, 20.0); ok = eb > 0
ax.errorbar(cen[ok], nb[ok] / eb[ok], yerr=np.sqrt(np.clip(nb[ok], 1, None)) / eb[ok], fmt="o", ms=2.5, color="k", lw=0.8, label="32S, 2026-05-08 (51k ions)")
for r, col, lab in ((r_sym, "C0", "symmetric"), (r_tail, "C3", "exponential tail")):
    lsf = rs.LineshapeFit(params=r["fit_params"], errors={}, deviance=0, n_data=0, n_free=0, success=True, message="", ripple_fixed=True,
                          x_mean=float(np.average(fx, weights=fe)), tail_model=r["fit_quality"].get("tail_model", "none"), tail_side=-1)
    mu = fe * rs.model_rate(lsf, fx, laser_fwhm_MHz=21.213)
    mb = np.bincount(idx, weights=mu, minlength=cen.size)
    dev = r["fit_quality"].get("deviance", r["fit_quality"].get("deviance_symmetric"))
    ax.plot(cen[ok], mb[ok] / eb[ok], color=col, lw=1.4, label=lab)
    axr.plot(cen[ok], (nb[ok] - mb[ok]) / np.sqrt(mb[ok]), ".-", color=col, lw=0.8, ms=3)
q = r_tail["fit_quality"]
ax.set_ylabel("ions per bunch"); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"(a) tail: f = {q['tail_fraction']:.2f}, decay {q['tail_length_V']:.1f} V; deviance {q['deviance_symmetric']:.0f} to {q['deviance']:.0f}", fontsize=9)
axr.axhline(0, color="0.5", lw=0.8); axr.set_ylabel("pull"); axr.set_xlabel("Doppler-corrected frequency (MHz)")
plt.setp(ax.get_xticklabels(), visible=False)

# (b) tail fraction vs CEC temperature
ax = fig.add_subplot(gs[0:2, 1])
rows = json.load(open(HERE / "survey_rows.json"))
col = {"Mar": "C0", "Apr": "C2", "May-Jun": "C3"}
for r in rows:
    if r["label"] != "32S" or r["counts"] < 2500 or not np.isfinite(r["cec"] or np.nan):
        continue
    e = min(r["f_e"], 0.15) if np.isfinite(r["f_e"]) and r["f_e"] > 0 else 0.0
    mk = {"library": "o", "power_0515": "s", "rf_0429": "v"}.get(r["group"], "^")
    ax.errorbar(r["cec"], r["f"], yerr=e, marker=mk, color=col[r["era"]], ls="none", ms=6, capsize=2)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="none", label=k) for k, c in col.items()] +
          [Line2D([], [], color="k", marker="s", ls="none", mfc="w", label="05-15 power series"),
           Line2D([], [], color="k", marker="v", ls="none", mfc="w", label="04-29 (532 nm NRI)")], fontsize=8, loc="lower left")
ax.set_xlabel("CEC reservoir temperature (C, PicoLog)"); ax.set_ylabel("tail fraction f (32S scans with at least 2500 ions)")
ax.set_title("(b) tail strength vs CEC temperature: no trend within a period", fontsize=9)
ax.set_ylim(0, 1.05)

# (c) collinear vs anticollinear: which side does the tail prefer
ax = fig.add_subplot(gs[0:2, 2])
ac = json.load(open(HERE / "anticollinear_test.json"))
names, vals, cols = [], [], []
for stamp, r in sorted(ac.items(), key=lambda kv: (kv[1]["geom"], kv[0])):
    d = r["high"]["dev"] - r["low"]["dev"]      # >0: tail on the LOW side preferred
    names.append(f"{stamp} {'C' if r['geom'] == 'collinear' else 'AC'}"); vals.append(d); cols.append("C0" if r["geom"] == "collinear" else "C1")
y = np.arange(len(names))
ax.barh(y, vals, color=cols)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.axvline(0, color="k", lw=0.8)
ax.set_xscale("symlog", linthresh=100)
ax.set_xlabel("deviance(tail high) - deviance(tail low)")
ax.set_title("(c) June 17/18 calibration scans: tail flips with geometry\n(collinear: low side; anticollinear: high side) = energy loss", fontsize=9)

# (d) F2 within-bunch test
ax = fig.add_subplot(gs[2, 0])
f2 = json.load(open(HERE / "bunch_f2.json"))["scan_20260508_140738.csv"]
ax.errorbar(f2["cen"], f2["F2"], yerr=f2["err"], fmt="o", ms=3, color="k", lw=0.8, label="data, 32S 2026-05-08")
ax.plot(f2["cen"], f2["intra"], color="C0", lw=1.5, label=f"loss within each bunch (chi2 {f2['chi_intra']:.0f})")
ax.plot(f2["cen"], f2["b2b"], color="C3", lw=1.5, label=f"whole bunches lose energy (chi2 {f2['chi_b2b']:.0f})")
ax.set_xlabel("Doppler-corrected frequency (MHz)"); ax.set_ylabel(r"$F_2=\langle k(k-1)\rangle/\langle k\rangle^2$ per bunch")
ax.legend(fontsize=8, loc="upper center"); ax.set_ylim(0.8, 7)
ax.set_title("(d) count statistics per bunch: the tail is within every bunch", fontsize=9)

# (e) Lorentzian vs 396 power
ax = fig.add_subplot(gs[2, 1])
pb_all = [r for r in json.load(open(HERE / "survey_rows.json")) if r["group"] == "power_0515"]
for r in pb_all:
    big = r["counts"] >= 2500
    ax.errorbar(r["p396"], r["gam"], yerr=min(r["gam_e"], 30) if np.isfinite(r["gam_e"]) else 0, marker="s", color="C3",
                mfc="C3" if big else "white", ls="none", capsize=2)
ax.legend(handles=[Line2D([], [], color="C3", marker="s", ls="none", label="at least 2500 ions"),
                   Line2D([], [], color="C3", marker="s", mfc="white", ls="none", label="fewer ions")], fontsize=8, loc="upper left")
ax.set_xscale("log"); ax.set_xlabel("396 nm power at CREMA entrance (mW)"); ax.set_ylabel("Lorentzian HWHM, tail model (MHz)")
ax.set_title("(e) 2026-05-15: power broadening only in the Lorentzian", fontsize=9)

# (f) IS per run group
ax = fig.add_subplot(gs[2, 2])
res = json.load(open(HERE / "is_final_summary.json"))
for k, (name, color, dx) in enumerate((("sym", "C0", -0.12), ("tail", "C3", 0.12))):
    g = res[name]["groups"]
    xs = np.arange(len(g)) + dx
    ax.errorbar(xs, [v["value_MHz"] for v in g], yerr=[v["sigma_MHz"] for v in g], fmt="o", color=color, capsize=2,
                label=f"{res[name]['label']}: {res[name]['mean']:.1f}({res[name]['unc']:.1f})")
    ax.axhspan(res[name]["mean"] - res[name]["unc"], res[name]["mean"] + res[name]["unc"], color=color, alpha=0.12)
ax.set_xticks(np.arange(len(res["sym"]["groups"])))
labs = [v["collection_date"][5:] for v in res["sym"]["groups"]]
labs = [l + (" br" if labs.count(l) > 1 and k == labs.index(l) else "") for k, l in enumerate(labs)]
ax.set_xticklabels(labs, fontsize=8, rotation=30)
ax.set_ylabel("34S-32S shift (MHz)"); ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(530, 650)
ax.set_title("(f) run groups, each with its own beam-energy calibration", fontsize=9)
fig.savefig(HERE / "tail_study_summary.png", dpi=110, bbox_inches="tight")
print("saved")
