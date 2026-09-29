"""Is the energy-loss tail within each bunch or bunch-to-bunch?  F2(x) = <k(k-1)>/<k>^2 across the line.

Per bunch: k = echo-cleaned in-gate hits, x = Doppler-corrected cell frequency. For a
rate lambda_b = A_b R_b(x) (A_b = bunch intensity, <A>=1, var v):
  intra-bunch tail   R_b(x) = R(x - s*eps_b), eps_b the 60 Hz ripple offset only
  bunch-to-bunch     R_b(x) = core(x - s*(eps_b - d_b)), d_b ~ (1-f) delta + f Exp(lambda)
F2(x) = (1+v) E[R_b^2]/E[R_b]^2; dead time scales it by a constant. Both shapes use the
fitted tail-model parameters; the overall constant is fitted to the line center.
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import quick_isotope_shift as qis, rate_spectrum as rs, ripple_lineshape as rl, isotope_shift_analysis as two_fit

D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"; U = r"C:\Users\EMALAB\data"
CASES = [(D + r"\scan_20260508_131731.csv", (4.25, 5.5)), (D + r"\scan_20260508_140738.csv", (4.25, 5.5)),
         (D + r"\scan_20260511_083642.csv", (4.25, 5.25)), (D + r"\scan_20260512_144505.csv", (4.25, 5.25)),
         (D + r"\scan_20260515_172054.csv", (4.25, 5.25)), (U + r"\scan_20260515_120957.csv", (4.25, 5.25))]
MASS = qis.SULFUR_MASSES_U["32S"]
opts = dict(qis.DEFAULT_ANALYSIS_OPTIONS); opts.update(voltage_offset_V=184.54201214242858, bootstrap_replicas=0, tail_model="exponential")
fig, axes = plt.subplots(len(CASES), 2, figsize=(14, 3.6 * len(CASES)))
rng = np.random.default_rng(3)
saved = {}
for row, (path, gate) in enumerate(CASES):
    spec, _ = qis._prepare_cut_file_for_label("32S", [Path(path)], options=opts, per_isotope_tof_gates={"32S": gate})
    fit = rs.fit_rate_spectrum(spec, MASS, opts)
    p = fit["fit_params"]; nu_ref = fit["nu_ref_GHz"] * 1000.0
    inp = rs._fit_inputs(spec, MASS, opts, nu_ref=nu_ref)
    s = inp["slope_MHz_per_V"]; a = p["ripple_halfwidth"]
    # per-bunch counts and Doppler-corrected frequency
    frame = rs.load_scan_frame([Path(path)], rs.echo_windows(opts))
    nu_lab = rs._lab_frequency_MHz(frame, opts); volt = rs._voltage_V(frame, opts) + opts["voltage_offset_V"]
    hits = rs.hit_mask(frame) & ~frame["is_echo"].to_numpy()
    tof = frame["tof"].to_numpy() * 1e6
    in_gate = hits & (tof > gate[0]) & (tof < gate[1])
    bunch, _ = pd.factorize(frame["bunch_id"].to_numpy())
    nb = bunch.max() + 1
    first = np.full(nb, -1); first[bunch[::-1]] = np.arange(bunch.size)[::-1]
    k = np.bincount(bunch[in_gate], minlength=nb).astype(float)
    x = two_fit.doppler_correct_ghz(nu_lab[first], MASS, volt[first], 1, "collinear") - nu_ref
    ok = np.isfinite(x)
    x, k = x[ok], k[ok]
    edges = np.arange(-900, 760, 20.0); cen = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(x, edges) - 1
    good = (idx >= 0) & (idx < cen.size)
    N = np.bincount(idx[good], minlength=cen.size).astype(float)
    S1 = np.bincount(idx[good], weights=k[good], minlength=cen.size)
    S2 = np.bincount(idx[good], weights=(k * (k - 1))[good], minlength=cen.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        F2 = N * S2 / S1 ** 2
        m = S1 / N
    # predictions (Monte Carlo over ripple phase and tail draws)
    eps = a * np.cos(2 * np.pi * rng.random(20000)) / s               # ripple offset in V (arcsine)
    d = np.where(rng.random(20000) < p["tail_fraction"], rng.exponential(p["tail_length"] / s, 20000), 0.0)
    live = np.average(spec.live, weights=spec.bunches)
    q_nt = dict(p, ripple_halfwidth=0.0)                               # full shape (core+tail) without ripple
    q_core = dict(p, ripple_halfwidth=0.0, tail_fraction=0.0)
    lw = dict(laser_fwhm_MHz=21.213)
    xx = cen
    grid = np.linspace(-3000, 3000, 12001)
    prof_nt = rs.line_profile(grid, q_nt, tail_model="exponential", **lw)
    prof_core = rs.line_profile(grid, q_core, tail_model="exponential", **lw)
    def moments(shift_V):
        # R_b(x) = b0 + amp*profile(x + s*shift) for each bunch; shift positive = energy loss? (loss -> lower x: profile(x + s*d))
        vals = []
        return None
    R_intra = p["background"] + p["amplitude"] * np.interp(xx[:, None] - s * eps[None, :2000], grid, prof_nt)
    shift = eps[:2000] - d[:2000]
    R_b2b = p["background"] + p["amplitude"] * np.interp(xx[:, None] - s * shift[None, :], grid, prof_core)
    pred_intra = np.mean(R_intra ** 2, 1) / np.mean(R_intra, 1) ** 2
    pred_b2b = np.mean(R_b2b ** 2, 1) / np.mean(R_b2b, 1) ** 2
    sel = (N > 200) & (S1 > 300)
    core_sel = sel & (np.abs(cen - p["center"]) < 120)
    c_intra = np.nanmedian(F2[core_sel] / pred_intra[core_sel]); c_b2b = np.nanmedian(F2[core_sel] / pred_b2b[core_sel])
    err = np.sqrt(np.clip(2 * S2, 1, None)) * N / S1 ** 2          # rough: Poisson on pair counts
    chi_i = np.nansum(((F2 - c_intra * pred_intra) / err)[sel] ** 2); chi_b = np.nansum(((F2 - c_b2b * pred_b2b) / err)[sel] ** 2)
    name = Path(path).name
    print(f"{name}: f {p['tail_fraction']:.2f} lamV {p['tail_length'] / s:.1f}  bins {sel.sum()}  chi2 intra-bunch {chi_i:7.1f}  bunch-to-bunch {chi_b:7.1f}"
          f"  (F2 core {np.nanmedian(F2[core_sel]):.3f}, far tail x<-400: obs {np.nanmean(F2[sel & (cen < -400)]):.3f} "
          f"intra {c_intra * np.mean(pred_intra[sel & (cen < -400)]):.3f} b2b {c_b2b * np.mean(pred_b2b[sel & (cen < -400)]):.3f})", flush=True)
    saved[name] = dict(cen=cen[sel].tolist(), F2=F2[sel].tolist(), err=err[sel].tolist(), intra=(c_intra * pred_intra[sel]).tolist(),
                       b2b=(c_b2b * pred_b2b[sel]).tolist(), chi_intra=float(chi_i), chi_b2b=float(chi_b))
    ax, ax2 = axes[row, 0], axes[row, 1]
    ax.errorbar(cen[sel], F2[sel], yerr=err[sel], fmt="o", ms=3, color="k", label="data")
    ax.plot(cen[sel], c_intra * pred_intra[sel], color="C0", label=f"tail within bunch (chi2 {chi_i:.0f})")
    ax.plot(cen[sel], c_b2b * pred_b2b[sel], color="C3", label=f"tail bunch-to-bunch (chi2 {chi_b:.0f})")
    ax.set_ylabel("F2 = <k(k-1)>/<k>^2"); ax.set_title(name); ax.legend(fontsize=8)
    ax2.plot(cen[sel], m[sel], "k.", label="<k> per bunch"); ax2.set_ylabel("ions/bunch"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.savefig("bunch_f2.png", dpi=80)
json.dump(saved, open("bunch_f2.json", "w"))
