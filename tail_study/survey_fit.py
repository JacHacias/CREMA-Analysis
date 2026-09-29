"""Width/shape survey: symmetric v2 and free exponential-tail fits of every survey scan."""
import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"
sys.path.insert(0, REPO)
HERE = Path(__file__).resolve().parent


def fwhm_numeric(f, lo=-3000.0, hi=3000.0, step=0.5):
    x = np.arange(lo, hi, step); y = f(x); m = y.max(); above = np.flatnonzero(y >= 0.5 * m)
    return float(x[above[-1]] - x[above[0]]) if above.size else float("nan")


def run(scan):
    import matplotlib; matplotlib.use("Agg")
    import quick_isotope_shift as qis, rate_spectrum as rs, ripple_lineshape as rl
    lab, path, gate = scan["label"], scan["path"], tuple(scan["gate"])
    mass = qis.SULFUR_MASSES_U[lab]
    base = dict(qis.DEFAULT_ANALYSIS_OPTIONS)
    base.update(voltage_offset_V=184.54201214242858, bootstrap_replicas=0, validate_isotope_wavenumber=False)
    t0 = time.time()
    spec, summary = qis._prepare_cut_file_for_label(lab, [Path(path)], options=base, per_isotope_tof_gates={lab: gate})
    rec = dict(scan)
    rec["counts"] = spec.num_points
    rec["diag"] = {k: v for k, v in spec.diagnostics.items() if k != "echo_windows_ns"}
    rec["scans_removed"] = summary.get("scans_removed"); rec["scans_total"] = summary.get("scans_total")
    tof = spec.hit_tof_s * 1e6
    rec["tof_mean_us"] = float(np.mean(tof)) if tof.size else None
    rec["tof_rms_us"] = float(np.std(tof)) if tof.size else None
    # all echo-cleaned hits per bunch (RFQ-load proxy) from the raw frame
    import rate_spectrum as rs2
    frame = rs2.load_scan_frame([Path(path)], rs2.echo_windows(base))
    hits = rs2.hit_mask(frame) & ~frame["is_echo"].to_numpy()
    rec["all_hits_per_bunch"] = float(hits.sum() / frame["bunch_id"].nunique())
    rec["fits"] = {}
    for name, extra in (("sym", {}), ("tail", {"tail_model": "exponential"})):
        opts = dict(base, **extra)
        try:
            r = rs.fit_rate_spectrum(spec, mass, opts)
        except Exception as exc:
            rec["fits"][name] = {"error": str(exc)}
            continue
        p, e, q = r["fit_params"], r["fit_errors"], r["fit_quality"]
        lw = dict(laser_fwhm_MHz=q["laser_fwhm_MHz"], laser_lineshape=q["laser_lineshape"])
        core = fwhm_numeric(lambda x: rl.ripple_voigt(x, p["sigma_doppler"], p["gamma"], 0.0, **lw))
        inp = rs._fit_inputs(spec, mass, opts, nu_ref=r["nu_ref_GHz"] * 1000.0)
        prof_params = dict(p)
        prof_params["ripple_halfwidth"] = 0.0
        # FWHM of the whole line without the ripple (core + tail), and with it
        tm = opts.get("tail_model", "none")
        whole_noripple = fwhm_numeric(lambda x: rs.line_profile(x + p["center"], prof_params, tail_model=tm, **lw))
        whole = fwhm_numeric(lambda x: rs.line_profile(x + p["center"], p, tail_model=tm, **lw))
        rec["fits"][name] = dict(
            params=p, errors=e, center_abs_MHz=r["center_abs_GHz"] * 1000.0, center_unc_MHz=r["center_fit_unc_GHz"] * 1000.0,
            deviance=float(q.get("deviance", float("nan"))) if "deviance" in q else None,
            deviance_per_dof=q["deviance_per_dof"], reduced_chi2=q["reduced_chi2"], n_free=None,
            slope_MHz_per_V=inp["slope_MHz_per_V"], core_fwhm_MHz=core, fwhm_noripple_MHz=whole_noripple, fwhm_MHz=whole,
            quality={k: v for k, v in q.items() if isinstance(v, (int, float, str, bool)) or v is None},
        )
    rec["elapsed_s"] = time.time() - t0
    return rec


if __name__ == "__main__":
    scans = json.load(open(HERE / "survey_cec.json"))
    only = set(sys.argv[1:])
    if only:
        scans = [s for s in scans if s["group"] in only or s["file"] in only]
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for rec in pool.map(run, scans):
            out.append(rec)
            fs, ft = rec["fits"].get("sym", {}), rec["fits"].get("tail", {})
            def g(f, k, sub="params"):
                try:
                    return f[sub][k]
                except Exception:
                    return float("nan")
            print(f"{rec['group']:10s} {rec['label']} {rec['file']:28s} n={rec['counts']:6d} "
                  f"sym: sd {g(fs,'sigma_doppler'):6.1f} gam {g(fs,'gamma'):5.1f} fwhm_nr {fs.get('fwhm_noripple_MHz', float('nan')):6.1f} | "
                  f"tail: sd {g(ft,'sigma_doppler'):6.1f} gam {g(ft,'gamma'):5.1f} f {g(ft,'tail_fraction'):.3f} lam {g(ft,'tail_length'):6.1f} "
                  f"dDev {(ft.get('quality',{}).get('deviance_symmetric') or float('nan')) - (ft.get('quality',{}).get('deviance') or float('nan')):7.1f} "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
    json.dump(out, open(HERE / ("survey_fit.json" if not only else f"survey_fit_{'_'.join(sorted(only))}.json"), "w"),
              indent=1, default=float)
