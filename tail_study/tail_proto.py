"""Prototype: exponential energy-loss tail on the ripple-averaged Voigt, high-stat 32S scans."""
import json, math, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quick_isotope_shift as qis, rate_spectrum as rs, ripple_lineshape as rl

D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
CASES = [("32S", D + r"\scan_20260508_131731.csv", (4.25, 5.5)), ("32S", D + r"\scan_20260508_140738.csv", (4.25, 5.5)),
         ("32S", D + r"\scan_20260511_083642.csv", (4.25, 5.25)), ("32S", D + r"\scan_20260512_144505.csv", (4.25, 5.25)),
         ("32S", D + r"\scan_20260508_153504.csv", (4.25, 5.5))]


def exp_tail2(h, p, lam):
    r = math.exp(-h / lam)
    c1 = (lam * (1.0 - r) - h * r) / h
    c0 = (1.0 - r) - c1
    out = np.empty_like(p)
    out[-1] = p[-1]
    # vectorized via lfilter on reversed sequence: y[m] = c0 p[m] + c1 p[m-1] + r y[m-1]
    pr = p[::-1]
    y = lfilter([c0, c1], [1.0, -r], pr)
    # lfilter assumes p[-1] = 0 and y[-1] = 0 before start; correct the start transient exactly:
    # homogeneous solution adds (T_N - y_0) r^m with the true T_N = p_N (flat continuation)
    y0_true = pr[0]
    corr = (y0_true - y[0]) * r ** np.arange(pr.size)
    return (y + corr)[::-1]


def make_model(fx, laser_fwhm, ripple_hw, h=1.0):
    lo, hi = fx.min() - 50.0, fx.max() + 50.0
    x_mean = None
    def profile(x, x0, sd, gam, f, lam):
        core = rl.ripple_voigt(x - x0, sd, gam, ripple_hw, laser_fwhm_MHz=laser_fwhm)
        if f <= 0 or lam <= 0:
            return core
        ext = min(10.0 * lam, 4000.0)
        grid = np.arange(lo, hi + ext + h, h)
        p = rl.ripple_voigt(grid - x0, sd, gam, ripple_hw, laser_fwhm_MHz=laser_fwhm)
        t = exp_tail2(h, p, lam)
        return (1.0 - f) * core + f * np.interp(x, grid, t)
    return profile


def fit(fx, fn, fe, profile, theta0, free, lower, upper, x_mean):
    names = ["amp", "x0", "sd", "gam", "b0", "b1", "f", "lam"]
    theta0 = np.array(theta0, float)
    free = np.array(free)
    def rate(th, x):
        amp, x0, sd, gam, b0, b1, f, lam = th
        return b0 + b1 * (x - x_mean) + amp * profile(x, x0, sd, gam, f, lam)
    def full(v):
        th = theta0.copy(); th[free] = v; return th
    def resid(v):
        mu = fe * np.clip(rate(full(v), fx), 1e-12, None)
        return rs._deviance_residuals(fn, mu)
    lo = np.array(lower, float)[free]; hi = np.array(upper, float)[free]
    x_start = np.clip(theta0[free], lo + 1e-6 * (np.abs(lo) + 1) * np.isfinite(lo), hi - 1e-6 * (np.abs(hi) + 1) * np.isfinite(hi))
    x_start = np.where(np.isfinite(x_start), x_start, theta0[free])
    res = least_squares(resid, x_start, bounds=(lo, hi), x_scale="jac", max_nfev=3000)
    th = full(res.x)
    mu = fe * np.clip(rate(th, fx), 1e-12, None)
    steps = np.maximum(1e-4 * np.abs(th), np.array([1e-4 * max(th[0], 1e-3), 0.05, 0.05, 0.05, 1e-6, 1e-9, 1e-4, 0.05]))
    at_b = (np.abs(res.x - lo) < 1e-6 * (np.abs(lo) + 1)) | (np.abs(res.x - hi) < 1e-6 * (np.abs(hi) + 1))
    act = free[~at_b]
    J = np.empty((mu.size, act.size))
    for k, j in enumerate(act):
        a, b = th.copy(), th.copy(); a[j] += steps[j]; b[j] -= steps[j]
        J[:, k] = (fe * np.clip(rate(a, fx), 1e-12, None) - fe * np.clip(rate(b, fx), 1e-12, None)) / (2 * steps[j])
    cov = np.linalg.pinv(J.T @ (J / mu[:, None]))
    err = dict.fromkeys(names, 0.0)
    for k, j in enumerate(act):
        err[names[j]] = math.sqrt(max(cov[k, k], 0))
    return dict(zip(names, th)), err, float(np.sum(res.fun ** 2)), rate, th


out = {}
fig, axes = plt.subplots(len(CASES), 2, figsize=(15, 4 * len(CASES)))
for row, (lab, path, gate) in enumerate(CASES):
    opts = dict(qis.DEFAULT_ANALYSIS_OPTIONS); opts["voltage_offset_V"] = 184.54201214242858; opts["bootstrap_replicas"] = 0
    spec, _ = qis._prepare_cut_file_for_label(lab, [qis.Path(path)], options=opts, per_isotope_tof_gates={lab: gate})
    mass = qis.SULFUR_MASSES_U[lab]
    base = rs.fit_rate_spectrum(spec, mass, opts)
    inp = rs._fit_inputs(spec, mass, opts, nu_ref=base["nu_ref_GHz"] * 1000.0)
    fx, fn, fe = inp["fx"], inp["fn"], inp["fe"]
    keep = fe > 0; fx, fn, fe = fx[keep], fn[keep], fe[keep]
    x_mean = float(np.average(fx, weights=fe))
    prof = make_model(fx, inp["laser_fwhm"], inp["ripple_hw"])
    p = base["fit_params"]
    lower = [0, fx.min(), 0, 0, -np.inf, -np.inf, 0, 5.0]
    upper = [np.inf, fx.max(), 2000, 2000, np.inf, np.inf, 0.95, 3000.0]
    th_sym = [p["amplitude"], p["center"], p["sigma_doppler"], p["gamma"], p["background"], p["slope"], 0.0, 100.0]
    sym = fit(fx, fn, fe, prof, th_sym, [0, 1, 2, 3, 4, 5], lower, upper, x_mean)
    results = {"sym": sym}
    best = None
    for f0, lam0 in ((0.2, 100.0), (0.3, 250.0), (0.15, 500.0)):
        th = list(th_sym); th[6] = f0; th[7] = lam0; th[1] = p["center"] + 30
        r = fit(fx, fn, fe, prof, th, [0, 1, 2, 3, 4, 5, 6, 7], lower, upper, x_mean)
        if best is None or r[2] < best[2]:
            best = r
    results["tail"] = best
    # tail, flat background (slope fixed 0)
    best0 = None
    for f0, lam0 in ((0.2, 100.0), (0.3, 250.0), (0.15, 500.0)):
        th = list(th_sym); th[6] = f0; th[7] = lam0; th[5] = 0.0; th[1] = p["center"] + 30
        r = fit(fx, fn, fe, prof, th, [0, 1, 2, 3, 4, 6, 7], lower, upper, x_mean)
        if best0 is None or r[2] < best0[2]:
            best0 = r
    results["tail_flat"] = best0
    name = qis.Path(path).name
    out[name] = {}
    print(f"\n{lab} {name}  counts {int(fn.sum())}  cells {fx.size}  ripple a={inp['ripple_hw']:.1f}")
    for key, (par, err, dev, rate, th) in results.items():
        out[name][key] = dict(params=par, errors=err, deviance=dev)
        print(f"  {key:10s} dev {dev:9.1f}  x0 {par['x0']:+7.1f}({err['x0']:.1f})  sd {par['sd']:6.1f}({err['sd']:.1f}) gam {par['gam']:5.1f}({err['gam']:.1f})"
              f"  b0 {par['b0']:.4f} b1 {par['b1']*1000:+.4f}/GHz  f {par['f']:.3f}({err['f']:.3f}) lam {par['lam']:6.1f}({err['lam']:.1f})")
    # plot data vs models (binned 20 MHz) + pulls
    centers, nb, eb, idx = rs._binned(fx, fn, fe, 20.0)
    ok = eb > 0
    ax, axr = axes[row, 0], axes[row, 1]
    ax.errorbar(centers[ok], nb[ok] / eb[ok], yerr=np.sqrt(np.clip(nb[ok], 1, None)) / eb[ok], fmt="o", ms=3, color="k")
    for key, col in (("sym", "C0"), ("tail", "C3"), ("tail_flat", "C2")):
        par, err, dev, rate, th = results[key]
        mu = fe * rate(th, fx)
        mb = np.bincount(idx, weights=mu, minlength=centers.size)
        ax.plot(centers[ok], mb[ok] / eb[ok], color=col, label=f"{key} dev {dev:.0f}")
        axr.plot(centers[ok], (nb[ok] - mb[ok]) / np.sqrt(np.clip(mb[ok], 1e-9, None)), color=col, marker=".", lw=1)
    ax.set_title(f"{lab} {name}"); ax.legend(fontsize=8); axr.axhline(0, color="gray"); axr.set_ylabel("pull (20 MHz bins)")
plt.tight_layout(); plt.savefig("tail_proto.png", dpi=90)
json.dump(out, open("tail_proto.json", "w"), indent=1, default=float)
