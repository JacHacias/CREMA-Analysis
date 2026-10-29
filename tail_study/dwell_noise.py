"""Is the pass-to-pass line motion source-intensity noise?

1. Measure the extra-Poisson, multiplicative dwell-to-dwell noise eta (rate x (1 + eta xi)) from the
   residuals of the full-scan tail fit, after a per-pass scale factor, and its time correlation.
2. Monte Carlo: same scan, same exposures, model rate x (1 + eta xi) with that noise -> Poisson; fit
   every pass with the frozen shape (as per_pass_tail.py "shape_fixed") and compare the pass-center
   scatter with the observed one.
Writes dwell_noise.json.
"""
import dataclasses, json, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import quick_isotope_shift as qis, rate_spectrum as rs

HERE = Path(__file__).resolve().parent
A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"; D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
CASES = [("32S", D + r"\scan_20260508_131731.csv", (4.25, 5.5)), ("32S", D + r"\scan_20260508_153504.csv", (4.25, 5.5)),
         ("32S", D + r"\scan_20260511_083642.csv", (4.25, 5.25)), ("32S", D + r"\scan_20260512_144505.csv", (4.25, 5.25)),
         ("32S", D + r"\scan_20260513_102135.csv", (4.25, 5.25)), ("32S", A + r"\32S_3-27-26.csv", (4.25, 5.5)),
         ("32S", D + r"\scan_20260601_144018.csv", (4.25, 5.5)),
         ("34S", D + r"\scan_20260512_155458.csv", (4.7, 5.4))]
N_MC = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rng = np.random.default_rng(7)
base = dict(qis.DEFAULT_ANALYSIS_OPTIONS); base.update(voltage_offset_V=190.7726576475665, bootstrap_replicas=0)


def dwell_index(spec):
    key = spec.group_pass.astype(np.int64) * 100000 + spec.group_bin.astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    return inv, u // 100000, u % 100000


def ar1(n, rho):
    x = np.empty(n); x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return x


def pass_centers(spec, mass, opts):
    out = []
    for p in np.unique(spec.group_pass):
        sub = spec.subset(np.flatnonzero(spec.group_pass == p))
        try:
            out.append(rs.fit_rate_spectrum(sub, mass, opts)["center_abs_GHz"] * 1e3)
        except Exception:
            out.append(np.nan)
    return np.array(out)


def main():
    results = {}
    for lab, path, gate in CASES:
        mass = qis.SULFUR_MASSES_U[lab]
        spec, _ = qis._prepare_cut_file_for_label(lab, [Path(path)], options=base, per_isotope_tof_gates={lab: gate})
        opts = dict(base, tail_model="exponential")
        if lab == "34S":  # transfer the 32S tail, as in the IS analysis
            ref, _ = qis._prepare_cut_file_for_label("32S", [Path(D + r"\scan_20260512_144505.csv")], options=base,
                                                     per_isotope_tof_gates={"32S": (4.25, 5.25)})
            rfit = rs.fit_rate_spectrum(ref, qis.SULFUR_MASSES_U["32S"], dict(opts, shape_transfer="tail"))
            opts = rs.transferred_shape_options(dict(opts, shape_transfer="tail"), [rfit])
        inputs = rs._fit_inputs(spec, mass, opts)
        fit = rs.fit_poisson_lineshape(inputs["fx"], inputs["fn"], inputs["fe"], **rs._lineshape_kwargs(inputs))
        rate = np.clip(rs.model_rate(fit, inputs["fx"], laser_fwhm_MHz=inputs["laser_fwhm"],
                                     laser_lineshape=inputs["laser_shape"]), 1e-12, None)
        mu = inputs["fe"] * rate
        n = inputs["fn"]
        # dwell level, with a per-pass scale (slow intensity per pass removed)
        inv, dpass, dbin = dwell_index(spec)
        nd = np.bincount(inv, weights=n); md = np.bincount(inv, weights=mu)
        scale = {p: nd[dpass == p].sum() / md[dpass == p].sum() for p in np.unique(dpass)}
        md_s = md * np.array([scale[p] for p in dpass])
        use = md_s > 20
        eta2 = np.sum((nd[use] - md_s[use]) ** 2 - nd[use]) / np.sum(md_s[use] ** 2)
        eta = float(np.sqrt(max(eta2, 0.0)))
        # by count level: is the excess multiplicative?
        levels = []
        for lo, hi in ((20, 80), (80, 250), (250, 1e9)):
            m = (md_s > lo) & (md_s <= hi)
            if m.sum() > 5:
                e2 = np.sum((nd[m] - md_s[m]) ** 2 - nd[m]) / np.sum(md_s[m] ** 2)
                levels.append((lo, int(m.sum()), float(np.sign(e2) * np.sqrt(abs(e2)))))
        # time correlation of the fractional residual (dwells in acquisition order = pass, bin)
        r = np.where(use, (nd - md_s) / np.sqrt(np.clip(md_s, 1, None) + (eta * md_s) ** 2), np.nan)
        rho = []
        for lag in (1, 2, 5):
            a, b = r[:-lag], r[lag:]
            ok = np.isfinite(a) & np.isfinite(b) & (dpass[:-lag] == dpass[lag:])
            rho.append(float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else float("nan"))
        rho1 = max(rho[0], 0.0)

        # frozen shape as per_pass_tail.py
        q = rs.fit_rate_spectrum(spec, mass, opts)
        fq = q["fit_quality"]
        frozen = dict(base, tail_model="exponential", tail_fraction=fq["tail_fraction"], tail_length_V=fq["tail_length_V"],
                      sigma_doppler_V=fq["sigma_doppler_V"], lorentzian_hwhm_MHz=q["fit_params"]["gamma"])
        obs = pass_centers(spec, mass, frozen)
        # Monte Carlo
        sims, sims_poisson = [], []
        for k in range(N_MC):
            for eta_k, store in ((eta, sims), (0.0, sims_poisson)):
                if eta_k == 0.0 and k >= max(N_MC // 3, 5):
                    continue
                xi = np.zeros(inv.max() + 1)
                for p in np.unique(dpass):
                    idx = np.flatnonzero(dpass == p)
                    xi[idx] = ar1(idx.size, rho1)
                factor = np.clip(1 + eta_k * xi[inv], 0, None)
                fake = dataclasses.replace(spec, counts=rng.poisson(mu * factor).astype(float))
                c = pass_centers(fake, mass, frozen)
                store.append(float(np.nanstd(c - np.nanmean(c), ddof=1)))
        res = dict(eta=eta, eta_by_level=levels, rho_lag_1_2_5=rho, obs_pass_sd=float(np.nanstd(obs, ddof=1)),
                   obs_centers=(obs - np.nanmean(obs)).tolist(), mc_pass_sd_mean=float(np.mean(sims)),
                   mc_pass_sd_rms=float(np.sqrt(np.mean(np.square(sims)))),
                   poisson_pass_sd_rms=float(np.sqrt(np.mean(np.square(sims_poisson)))),
                   deviance_per_dof=fq["deviance_per_dof"])
        results[Path(path).name] = res
        print(f"{lab} {Path(path).name:26s} eta {eta:.3f} (by level {[(l, n_, round(e, 3)) for l, n_, e in levels]})"
              f" rho {[round(x, 2) for x in rho]} | pass sd obs {res['obs_pass_sd']:5.1f}  MC(eta) {res['mc_pass_sd_rms']:5.1f}"
              f"  MC(Poisson) {res['poisson_pass_sd_rms']:5.1f} MHz", flush=True)
    json.dump(results, open(HERE / "dwell_noise.json", "w"), indent=1)


if __name__ == "__main__":
    main()
