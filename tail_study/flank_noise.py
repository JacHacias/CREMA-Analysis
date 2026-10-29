"""Intensity noise or line (energy/frequency) jitter? Model-free test.

For each scan, rates per (pass, 20 MHz cell) after a per-pass scale are compared with the mean over
passes at the same cell. Intensity noise adds a fractional variance eta_I^2 everywhere; a line jitter
of sd delta adds delta^2 (d ln R / dx)^2, i.e. only on the flanks. Fit
    excess fractional variance = eta_I^2 + delta^2 * (d ln R/dx)^2
over cells with enough signal. Writes flank_noise.json.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quick_isotope_shift as qis, rate_spectrum as rs
from dwell_noise import CASES, base

HERE = Path(__file__).resolve().parent
CELL = 20.0
out = {}
pool_x, pool_y, pool_w = [], [], []
for lab, path, gate in CASES:
    mass = qis.SULFUR_MASSES_U[lab]
    spec, _ = qis._prepare_cut_file_for_label(lab, [Path(path)], options=base, per_isotope_tof_gates={lab: gate})
    inp = rs._fit_inputs(spec, mass, base)
    x, n, e = inp["fx"], inp["fn"], inp["fe"]
    cell = np.floor(x / CELL).astype(int)
    passes = np.unique(spec.group_pass)
    cells = np.unique(cell)
    N = np.zeros((passes.size, cells.size)); E = np.zeros_like(N)
    for i, p in enumerate(passes):
        m = spec.group_pass == p
        N[i] = np.bincount(np.searchsorted(cells, cell[m]), weights=n[m], minlength=cells.size)
        E[i] = np.bincount(np.searchsorted(cells, cell[m]), weights=e[m], minlength=cells.size)
    full = (E > 0).all(axis=0)
    N, E, xc = N[:, full], E[:, full], (cells[full] + 0.5) * CELL
    Rbar = N.sum(0) / E.sum(0)
    scale = (N.sum(1) / (E * Rbar).sum(1))[:, None]          # per-pass intensity
    mu = E * Rbar * scale
    # smooth log-derivative of the mean profile
    sm = np.convolve(Rbar, np.ones(3) / 3, mode="same")
    dlog = np.gradient(np.log(np.clip(sm, 1e-9, None)), xc)
    bg = np.percentile(Rbar, 10)
    sig_frac = np.clip((Rbar - bg) / Rbar, 0, 1)
    # excess fractional variance per cell (unbiased-ish: subtract Poisson, scale by k/(k-1))
    k = N.shape[0]
    frac = ((N - mu) ** 2 - N).sum(0) / (mu ** 2).sum(0) * k / (k - 1)
    use = (mu.mean(0) > 15) & (sig_frac > 0.5) & np.isfinite(dlog)
    X = np.c_[np.ones(use.sum()), dlog[use] ** 2]
    w = (mu.mean(0)[use]) ** 2 / (1 + mu.mean(0)[use] * 0.05)  # rough weights, high counts more precise
    coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], frac[use] * np.sqrt(w), rcond=None)
    eta_I = float(np.sign(coef[0]) * np.sqrt(abs(coef[0])))
    delta = float(np.sign(coef[1]) * np.sqrt(abs(coef[1])))
    # flank vs peak comparison in plain numbers
    s = np.abs(dlog[use])
    lo, hi = s < np.percentile(s, 33), s > np.percentile(s, 67)
    fl = lambda m: float(np.sqrt(max(np.sum(w[m] * frac[use][m]) / np.sum(w[m]), 0)))
    out[Path(path).name] = dict(iso=lab, passes=int(k), eta_intensity=eta_I, delta_MHz=delta,
                                frac_sd_peak=fl(lo), frac_sd_flank=fl(hi),
                                dlog_peak=float(np.median(s[lo])), dlog_flank=float(np.median(s[hi])))
    pool_x += list(dlog[use] ** 2); pool_y += list(frac[use]); pool_w += list(w)
    print(f"{lab} {Path(path).name:26s} passes {k}: intensity eta {eta_I:+.3f}, line jitter delta {delta:+6.1f} MHz"
          f" | frac sd near peak {fl(lo):.3f} (|dlnR/dx| {np.median(s[lo]):.4f}/MHz) vs flanks {fl(hi):.3f}"
          f" ({np.median(s[hi]):.4f}/MHz)", flush=True)

px, py, pw = map(np.array, (pool_x, pool_y, pool_w))
X = np.c_[np.ones(px.size), px]
coef, *_ = np.linalg.lstsq(X * np.sqrt(pw)[:, None], py * np.sqrt(pw), rcond=None)
out["pooled"] = dict(eta_intensity=float(np.sqrt(max(coef[0], 0))), delta_MHz=float(np.sqrt(max(coef[1], 0))))
print(f"\npooled: intensity eta {out['pooled']['eta_intensity']:.3f}, line jitter delta {out['pooled']['delta_MHz']:.1f} MHz")
json.dump(out, open(HERE / "flank_noise.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(5, 3.6))
order = np.argsort(px)
bins = np.quantile(px, np.linspace(0, 1, 9))
idx = np.clip(np.digitize(px, bins) - 1, 0, 7)
bx = [np.sqrt(np.average(px[idx == i], weights=pw[idx == i])) for i in range(8)]
by = [np.average(py[idx == i], weights=pw[idx == i]) for i in range(8)]
ax.plot(np.sqrt(px), py, ".", ms=3, alpha=0.3, color="0.5")
ax.plot(bx, by, "o", color="C0", label="binned")
xx = np.linspace(0, np.sqrt(px.max()), 100)
ax.plot(xx, coef[0] + coef[1] * xx ** 2, "-", color="C3",
        label=f"$\\eta_I$={out['pooled']['eta_intensity']:.2f}, $\\delta$={out['pooled']['delta_MHz']:.0f} MHz")
ax.set_xlabel(r"$|d\,\ln R/d\nu|$ (1/MHz)"); ax.set_ylabel("excess fractional variance")
ax.set_ylim(-0.05, 0.4); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "flank_noise.png", dpi=150)
