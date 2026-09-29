"""Correlate fitted widths / tail with CEC temperature, 396 power, ion rate, ablation, era."""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "survey_fit.json"))


def era(r):
    d = r["start"][:10]
    return "Mar" if d < "2026-04-01" else ("Apr" if d < "2026-05-01" else "May-Jun")


rows = []
for r in S:
    fs, ft = r["fits"].get("sym", {}), r["fits"].get("tail", {})
    if "params" not in fs or "params" not in ft:
        continue
    ps, pt, et, qt = fs["params"], ft["params"], ft["errors"], ft["quality"]
    s = ft["slope_MHz_per_V"]
    row = dict(file=r["file"], label=r["label"], group=r["group"], era=era(r), counts=r["counts"],
               cec=r.get("cec_mean"), cec_slope=r.get("cec_slope_per_h"), p396=r["cond"].get("P396"),
               nri=r["cond"].get("NRI"), abl=r["cond"].get("ABL"), iris=r["cond"].get("iris"),
               rate=r["diag"]["peak_dwell_rate_per_bunch"], hits_pb=r["all_hits_per_bunch"],
               tof_rms=r["tof_rms_us"], start=r["start"],
               sym_fwhm=fs["fwhm_noripple_MHz"], sym_sd=ps["sigma_doppler"], sym_gam=ps["gamma"],
               f=pt["tail_fraction"], f_e=et["tail_fraction"], lamV=pt["tail_length"] / s, lamV_e=et["tail_length"] / s,
               sdV=pt["sigma_doppler"] / s, sdV_e=et["sigma_doppler"] / s, gam=pt["gamma"], gam_e=et["gamma"],
               core_fwhm=ft["core_fwhm_MHz"], tail_fwhm=ft["fwhm_noripple_MHz"],
               ddev=(qt.get("deviance_symmetric") or np.nan) - (qt.get("deviance") or np.nan), slope=s,
               bg_slope=qt.get("background_slope_per_GHz"))
    row["meanloss_V"] = row["f"] * row["lamV"]
    rows.append(row)

good = [r for r in rows if r["counts"] >= 2500]
print(f"{len(rows)} fitted scans, {len(good)} with >= 2500 in-gate ions\n")
hdr = f"{'file':28s} {'lab':3s} {'era':7s} {'n':>6s} {'CEC':>6s} {'P396':>5s} {'rate':>5s} {'symFWHM':>7s} | {'f':>5s} {'lamV':>5s} {'<dE>V':>5s} {'sdV':>5s} {'gam':>5s} {'dDev':>6s}"
print(hdr)
for r in sorted(good, key=lambda r: r["start"]):
    print(f"{r['file']:28s} {r['label']:3s} {r['era']:7s} {r['counts']:6d} {r['cec'] or float('nan'):6.1f} {r['p396'] if r['p396'] is not None else float('nan'):5.2f} "
          f"{r['rate']:5.2f} {r['sym_fwhm']:7.1f} | {r['f']:5.2f} {r['lamV']:5.1f} {r['meanloss_V']:5.1f} {r['sdV']:5.2f} {r['gam']:5.1f} {r['ddev']:6.0f}")


def spearman(sel, xk, yk, label):
    x = np.array([r[xk] for r in sel], float); y = np.array([r[yk] for r in sel], float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return
    rho, p = stats.spearmanr(x[ok], y[ok])
    pr, pp = stats.pearsonr(x[ok], y[ok])
    print(f"  {label:44s} {yk:>10s} vs {xk:<9s} n={ok.sum():2d}  Spearman {rho:+.2f} (p={p:.3f})  Pearson {pr:+.2f} (p={pp:.3f})")


lib32 = [r for r in good if r["group"] == "library" and r["label"] == "32S"]
lib_all = [r for r in good if r["group"] == "library"]
may32 = [r for r in good if r["label"] == "32S" and r["era"] == "May-Jun" and r["group"] == "library"]
pb = [r for r in good if r["group"] == "power_0515"]
all32 = [r for r in good if r["label"] == "32S"]
print("\nCorrelations")
for sel, name in ((lib32, "library 32S"), (lib_all, "library 32S+34S (>=2500 ions)"), (may32, "library 32S, May-Jun only"),
                  (all32, "all 32S (library+April+power series)")):
    for yk in ("f", "meanloss_V", "lamV", "sdV", "gam", "sym_fwhm", "tail_fwhm"):
        for xk in ("cec", "rate", "abl"):
            spearman(sel, xk, yk, name)
print("\n396 nm power series (2026-05-15, CEC 215 C, 32S):")
for yk in ("gam", "sdV", "f", "lamV", "meanloss_V", "sym_fwhm", "sym_gam", "tail_fwhm"):
    spearman(pb, "p396", yk, "power series")

# era comparison (32S, all groups)
print("\nEra means (32S scans >= 2500 ions):")
for e in ("Mar", "Apr", "May-Jun"):
    sel = [r for r in all32 if r["era"] == e]
    if not sel:
        continue
    arr = lambda k: np.array([r[k] for r in sel], float)
    print(f"  {e:8s} n={len(sel):2d}  f {arr('f').mean():.2f}+/-{arr('f').std(ddof=1) / math.sqrt(len(sel)):.2f}  "
          f"<dE> {arr('meanloss_V').mean():.1f} V  lamV {np.median(arr('lamV')):.1f}  sdV {arr('sdV').mean():.2f}  "
          f"gam {arr('gam').mean():.1f}  symFWHM {arr('sym_fwhm').mean():.0f}  dDev/1000ions {np.mean(arr('ddev') / arr('counts') * 1000):.1f}  CEC {np.nanmean(arr('cec')):.1f}")
mar = [r["f"] for r in all32 if r["era"] == "Mar"]; later = [r["f"] for r in all32 if r["era"] != "Mar"]
print(f"  Mann-Whitney f(Mar) vs f(Apr-Jun): p = {stats.mannwhitneyu(mar, later).pvalue:.4f}")

# Figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
col = {"Mar": "C0", "Apr": "C2", "May-Jun": "C3"}
mk = {"library": "o", "power_0515": "s", "rf_0428": "^", "rf_0429": "v"}
for r in good:
    kw = dict(color=col[r["era"]], marker=mk[r["group"]], ms=7 if r["label"] == "32S" else 5,
              mfc=col[r["era"]] if r["label"] == "32S" else "white", ls="none")
    ax[0, 0].errorbar(r["cec"], r["f"], yerr=r["f_e"], **kw)
    ax[0, 1].errorbar(r["cec"], r["meanloss_V"], **kw)
    ax[0, 2].errorbar(r["cec"], r["sdV"], yerr=r["sdV_e"], **kw)
    ax[1, 0].errorbar(r["rate"], r["f"], yerr=r["f_e"], **kw)
    ax[1, 1].errorbar(r["rate"], r["sdV"], yerr=r["sdV_e"], **kw)
for r in pb:
    ax[1, 2].errorbar(r["p396"], r["gam"], yerr=r["gam_e"], color="C3", marker="s", ls="none", label=None)
    ax[1, 2].errorbar(r["p396"], r["sym_gam"], color="0.6", marker="x", ls="none")
ax[0, 0].set(xlabel="CEC reservoir T (C, PicoLog)", ylabel="tail fraction f")
ax[0, 1].set(xlabel="CEC reservoir T (C)", ylabel="mean extra energy loss f*lambda (V)")
ax[0, 2].set(xlabel="CEC reservoir T (C)", ylabel="core Gaussian sigma (V of beam energy)")
ax[1, 0].set(xlabel="peak rate (ions/bunch)", ylabel="tail fraction f")
ax[1, 1].set(xlabel="peak rate (ions/bunch)", ylabel="core Gaussian sigma (V)")
ax[1, 2].set(xlabel="396 nm power (mW)", ylabel="Lorentzian HWHM (MHz)", xscale="log", title="2026-05-15 power series (x: symmetric fit)")
from matplotlib.lines import Line2D
ax[0, 0].legend(handles=[Line2D([], [], color=c, marker="o", ls="none", label=e) for e, c in col.items()] +
                [Line2D([], [], color="k", marker=m, ls="none", mfc="w", label=g) for g, m in mk.items()], fontsize=8)
plt.tight_layout(); plt.savefig(HERE / "survey_corr.png", dpi=90)
json.dump(rows, open(HERE / "survey_rows.json", "w"), indent=1, default=float)
