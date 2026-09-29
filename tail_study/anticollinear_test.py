"""Which side is the tail on in anticollinear geometry? Energy loss flips it, a laser artefact does not.

Fits the June 17/18 beam-energy calibration scans in the LAB frame (v2 cells, 32S) with
(a) the symmetric model, (b) the exponential tail at lower lab frequency, (c) at higher.
Collinear: an energy loss puts the tail LOW; anticollinear: HIGH. Then recomputes the
collinear/anticollinear beam-energy offset from each set of line centers.
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quick_isotope_shift as qis, rate_spectrum as rs, ripple_lineshape as rl, isotope_shift_analysis as isa

D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
SCANS = {"111137": ("20260617", "collinear"), "123956": ("20260617", "collinear"), "171758": ("20260617", "anticollinear"),
         "172745": ("20260617", "anticollinear"), "095024": ("20260618", "anticollinear"), "110246": ("20260618", "collinear"),
         "142931": ("20260618", "collinear"), "153348": ("20260618", "anticollinear"), "163309": ("20260618", "collinear"),
         "171531": ("20260618", "collinear")}
PAIRS = [(["111137", "123956"], ["171758", "172745"]), (["110246"], ["095024"]), (["142931"], ["153348"]),
         (["163309"], ["153348"]), (["171531"], ["153348"])]
MASS = qis.SULFUR_MASSES_U["32S"]

opts = dict(qis.DEFAULT_ANALYSIS_OPTIONS)
opts.update(voltage_offset_V=0.0, bootstrap_replicas=0, validate_isotope_wavenumber=False)
fits = {}
fig, axes = plt.subplots(len(SCANS), 1, figsize=(11, 3.2 * len(SCANS)))
for ax, (stamp, (day, geom)) in zip(axes, SCANS.items()):
    path = Path(D) / f"scan_{day}_{stamp}.csv"
    spec, _ = qis._prepare_cut_file_for_label("32S", [path], options=opts, per_isotope_tof_gates={"32S": (4.25, 5.5)})
    ref = float(np.median(spec.nu_lab_MHz))
    x = spec.nu_lab_MHz - ref
    fe = spec.bunches * spec.live
    fn = spec.counts
    v_mean = float(np.average(spec.voltage_V, weights=spec.bunches))
    a = rl.ripple_halfwidth_MHz(ref, MASS, v_mean + 184.54, 4.5, geometry=geom)
    kw = dict(ripple_halfwidth=a, laser_fwhm_MHz=21.213)
    sym = rs.fit_poisson_lineshape(x, fn, fe, **kw)
    lo = rs.fit_poisson_lineshape(x, fn, fe, **kw, tail_model="exponential", tail_side=-1)
    hi = rs.fit_poisson_lineshape(x, fn, fe, **kw, tail_model="exponential", tail_side=+1)
    rec = dict(geom=geom, n=int(fn.sum()), ref=ref, v_set=v_mean, a=a,
               sym=dict(c=sym.params["center"], e=sym.errors["center"], dev=sym.deviance),
               low=dict(c=lo.params["center"], e=lo.errors["center"], dev=lo.deviance, f=lo.params["tail_fraction"], lam=lo.params["tail_length"], sd=lo.params["sigma_doppler"]),
               high=dict(c=hi.params["center"], e=hi.errors["center"], dev=hi.deviance, f=hi.params["tail_fraction"], lam=hi.params["tail_length"], sd=hi.params["sigma_doppler"]))
    fits[stamp] = rec
    print(f"{day}_{stamp} {geom:13s} n={rec['n']:6d}  dev sym {sym.deviance:8.1f} | tail LOW {lo.deviance:8.1f} (f {lo.params['tail_fraction']:.2f}, lam {lo.params['tail_length']:6.1f}) "
          f"| tail HIGH {hi.deviance:8.1f} (f {hi.params['tail_fraction']:.2f}, lam {hi.params['tail_length']:6.1f})  -> preferred: "
          f"{'LOW' if lo.deviance < hi.deviance else 'HIGH'} by {abs(lo.deviance - hi.deviance):.1f}", flush=True)
    centers, nb, eb, idx = rs._binned(x, fn, fe, 20.0)
    ok = eb > 0
    ax.errorbar(centers[ok], nb[ok] / eb[ok], yerr=np.sqrt(np.clip(nb[ok], 1, None)) / eb[ok], fmt="o", ms=3, color="k")
    for f, col, lab in ((sym, "C0", "sym"), (lo, "C3", "tail low"), (hi, "C2", "tail high")):
        mu = rs.model_rate(f, x, laser_fwhm_MHz=21.213)
        mb = np.bincount(idx, weights=fe * mu, minlength=centers.size)
        ax.plot(centers[ok], mb[ok] / eb[ok], color=col, label=f"{lab} dev {f.deviance:.0f}")
    ax.set_title(f"{day}_{stamp} {geom} (lab frame)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("anticollinear_test.png", dpi=80)


def beam_voltage(nu_c, nu_a):
    beta = (nu_c - nu_a) / (nu_c + nu_a)
    from scipy.optimize import brentq
    f = lambda v: float(isa.beam_beta_after_cec(MASS, beam_voltage_V=v, charge_e=1, neutralization="none")) - beta
    return brentq(f, 1.0, 5.0e6)


print("\nBeam-energy offset (inferred - set, V) per calibration pair:")
for model_c, model_a, name in (("sym", "sym", "symmetric"), ("low", "high", "energy tail (low in C, high in AC)"),
                               ("low", "low", "laser tail (low in both)")):
    offs = []
    for cols, antis in PAIRS:
        nu_c = np.mean([fits[s]["ref"] + fits[s][model_c]["c"] for s in cols])
        nu_a = np.mean([fits[s]["ref"] + fits[s][model_a]["c"] for s in antis])
        v_set = np.median([fits[s]["v_set"] for s in cols + antis])
        offs.append(beam_voltage(nu_c, nu_a) - v_set)
    print(f"  {name:36s} " + "  ".join(f"{o:7.2f}" for o in offs) + f"   mean {np.mean(offs):.2f} +/- {np.std(offs, ddof=1) / math.sqrt(len(offs)):.2f}")
json.dump(fits, open("anticollinear_test.json", "w"), indent=1)
