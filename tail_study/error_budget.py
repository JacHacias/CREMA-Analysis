"""Error budget for a 36S charge radius calibrated on muonic 34S-32S, and accumulation-time
scaling from the adopted 34S-32S library (energy-loss-tail analysis, 2026-09-29).

Writes error_budget.json. Run with ..\\.venv\\Scripts\\python.exe from this folder.
"""
import csv
import json
import math
import os
import re
from datetime import datetime

LIB = "../hfs_gui/data_library/isotope_shift_library.jsonl"
BUNCH_HZ = 50.0

# Atomic masses (u) and calculated atomic factors used in the thesis
M = {32: 31.97207117, 34: 33.96786701, 36: 35.96708071}
F = -47.9          # MHz/fm^2
K_MS = -327.9e3    # MHz u
DIS_DV_34 = 0.919  # MHz/V, measured beam-energy sensitivity of the 34-32 shift

# Muonic atoms, Schaller et al. PRC 31, 1007 (1985): rms radius differences (am = 1e-3 fm)
R32 = 3.2611
DR_34_32, DR_34_32_U = 0.0231, 0.0012
DR_36_34, DR_36_34_U = 0.0130, 0.0012


def mu(a, b):
    return (M[a] - M[b]) / (M[a] * M[b])


def scan_seconds(path):
    """Wall-clock seconds of one scan: filename start to file mtime for DAQ scans, bunch-id
    span at 50 Hz for the March archive files (no start time in the name)."""
    name = os.path.basename(path)
    m = re.match(r"scan_(\d{8}_\d{6})\.csv", name)
    if m and os.path.exists(path):
        start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
        return os.path.getmtime(path) - start, "wall"
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        first = last = None
        for row in rd:
            b = int(row[-1])
            first = b if first is None else first
            last = b
    return (last - first) / BUNCH_HZ, "bunch-span"


def main():
    out = {}
    # ---- muonic anchors -------------------------------------------------------------
    r34 = R32 + DR_34_32
    r36 = r34 + DR_36_34
    d34 = DR_34_32 * (R32 + r34)
    d34_u = DR_34_32_U * (R32 + r34)
    d3634 = DR_36_34 * (r34 + r36)
    d3634_u = DR_36_34_U * (r34 + r36)
    d36 = d34 + d3634
    d36_u = math.hypot(d34_u, d3634_u)  # assumes the two differences independent
    ratio = mu(36, 32) / mu(34, 32)
    out["muonic"] = dict(dr2_34_32=d34, dr2_34_32_unc=d34_u, dr2_36_34=d3634,
                         dr2_36_34_unc=d3634_u, dr2_36_32=d36, dr2_36_32_unc=d36_u,
                         mass_factor_ratio_36_34=ratio)

    # The K_MS-free combination the optical data measure:
    #   C = dnu36 - ratio*dnu34 = F * (dr2_36 - ratio*dr2_34)
    comb_muonic = d36 - ratio * d34
    comb_muonic_u = math.hypot(d3634_u, (ratio - 1) * d34_u)
    out["kms_free_combination"] = dict(
        definition="(dnu36 - r*dnu34)/F = dr2(36,32) - r*dr2(34,32) = dr2(36,34) - (r-1)*dr2(34,32)",
        muonic_fm2=comb_muonic, muonic_unc_fm2=comb_muonic_u,
        muonic_in_MHz=F * comb_muonic, muonic_unc_in_MHz=abs(F) * comb_muonic_u)

    # Beam-energy sensitivity of each shift scales with m_A^-1/2 - m_32^-1/2
    s = lambda a: M[a] ** -0.5 - M[32] ** -0.5
    dv36 = DIS_DV_34 * s(36) / s(34)
    out["beam_energy"] = dict(dIS34_dV=DIS_DV_34, dIS36_dV=dv36,
                              d_comb_dV=dv36 - ratio * DIS_DV_34)

    # Mass shifts from the calculated K_MS (for the theory-only route)
    out["theory_route"] = dict(mass_shift_34=K_MS * mu(34, 32), mass_shift_36=K_MS * mu(36, 32))

    # ---- library: time per pair and the resulting shift error ------------------------
    rows = [json.loads(line) for line in open(LIB)]
    adopted = []
    for r in rows:
        if r["comparison"] != "34S-32S" or "[excluded]" in (r["notes"] or ""):
            continue
        if "background" in r["run_label"] or "boundary" in r["run_label"]:
            continue  # not among the 7 adopted run groups
        fq = json.loads(r["bad_scan_filter"])["fit_quality"]
        pb = fq.get("pair_bootstrap", {})
        files = r["files"].split(";")
        secs, kinds = [], []
        for f in files:
            t, k = scan_seconds(f)
            secs.append(t)
            kinds.append(k)
        c = [fq[k]["counting_center_unc_MHz"] for k in fq if k.startswith(("32S", "34S"))]
        hits34 = fq["34S"]["in_gate_hits"]
        passes34 = fq["34S"]["bootstrap_units_per_stratum"]
        count = math.sqrt(sum(x * x for x in c[:2])) if len(c) == 2 else None
        if "32S_before" in fq:
            f = 0.5
            count = math.sqrt(((1 - f) * fq["32S_before"]["counting_center_unc_MHz"]) ** 2
                              + (f * fq["32S_after"]["counting_center_unc_MHz"]) ** 2
                              + fq["34S"]["counting_center_unc_MHz"] ** 2)
        total = r["isotope_shift_fit_unc_MHz"]
        adopted.append(dict(
            run=r["run_label"], shift=r["isotope_shift_MHz"], unc=total,
            pair_boot=pb.get("shift_unc_MHz"), counting=count,
            excess=math.sqrt(max(total ** 2 - count ** 2, 0.0)),
            hours=sum(secs) / 3600, time_kind=kinds, hours_34=secs[-1] / 3600 if len(files) == 2 else sum(secs[1:2]) / 3600,
            hits34=hits34, passes34=passes34,
            figure_of_merit=total * math.sqrt(sum(secs) / 3600)))
    out["pairs"] = adopted

    # Combination as adopted: 7 run groups (05-08 pairs 1/2/5 enter as one group)
    groups = json.load(open("final_numbers.json"))["tail"]
    sig = groups["stat"]
    hours = sum(p["hours"] for p in adopted)
    # 06-01 bracket and 06-01 share the 144018 32S scan: count it once
    shared = [p for p in adopted if p["run"].startswith("sulfur_20260601")]
    if len(shared) == 2:
        t, _ = scan_seconds(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data\scan_20260601_144018.csv")
        hours -= t / 3600
    # 05-08 pair 2 shares its 32S? (134603 is 34S in pair 1 and 32S in pair 2: count once)
    t, _ = scan_seconds(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data\scan_20260508_134603.csv")
    hours -= t / 3600
    K = sig * math.sqrt(hours)
    counting_only = 1 / math.sqrt(sum(1 / p["counting"] ** 2 for p in adopted))
    out["combination"] = dict(stat_MHz=sig, chi2r=groups["chi2r"], beam_hours=hours,
                              K_MHz_sqrt_h=K, counting_only_MHz=counting_only,
                              hours_for={str(x): (K / x) ** 2 for x in (3.0, 2.0, 1.5, 1.0, 0.5)})

    # ---- 36S scenarios ----------------------------------------------------------------
    def comb_unc(s34, s36, wm, rho_wm):
        """MHz error on C = dnu36 - r*dnu34 (stat + wavemeter with correlation rho)."""
        wm2 = wm ** 2 * (1 + ratio ** 2 - 2 * ratio * rho_wm)
        return math.sqrt(s36 ** 2 + (ratio * s34) ** 2 + wm2)

    scen = []
    for s34 in (4.9, 1.5, 1.0):
        for s36 in (10.0, 3.0, 1.0):
            for wm, rho in ((1.8, 0.0), (1.8, 1.0), (0.2, 0.0)):
                cu = comb_unc(s34, s36, wm, rho)
                ours = cu / abs(F)
                d36_ours = math.hypot(ours, ratio * d34_u)
                d3634_ours = math.hypot(ours, (ratio - 1) * d34_u)
                scen.append(dict(s34=s34, s36=s36, wavemeter=wm, rho=rho, comb_MHz=cu,
                                 comb_fm2=ours, dr2_36_32_unc=d36_ours,
                                 dr2_36_34_unc=d3634_ours))
    out["scenarios"] = scen

    # No 36S data from before October 2026 is trusted, so the 36S counting error is projected
    # from the 34S line (same line shape): sigma36 = k / sqrt(in-gate ions).
    k34 = [p["counting"] for p in adopted]
    out["k_34_count"] = [json.loads(r["bad_scan_filter"])["fit_quality"]["34S"]["counting_center_unc_MHz"]
                         * math.sqrt(json.loads(r["bad_scan_filter"])["fit_quality"]["34S"]["in_gate_hits"])
                         for r in rows if r["comparison"] == "34S-32S"]
    k = sorted(out["k_34_count"])[len(out["k_34_count"]) // 2]
    out["ions_36_needed"] = dict(k_MHz=k, **{str(x): (k / x) ** 2 for x in (10.0, 3.0, 1.0, 0.6)})
    json.dump(out, open("error_budget.json", "w"), indent=1)

    # ---- print ------------------------------------------------------------------------
    mu_ = out["muonic"]
    print(f"muonic dr2(34,32) = {d34:.4f}({d34_u*1e4:.0f}) fm2, dr2(36,34) = {d3634:.4f}({d3634_u*1e4:.0f}),"
          f" dr2(36,32) = {d36:.4f}({d36_u*1e4:.0f}); r = {ratio:.4f}")
    km = out["kms_free_combination"]
    print(f"K_MS-free combination, muonic: {km['muonic_fm2']:.4f}({km['muonic_unc_fm2']*1e4:.0f}) fm2"
          f" = {km['muonic_in_MHz']:.2f}({km['muonic_unc_in_MHz']:.2f}) MHz")
    be = out["beam_energy"]
    print(f"beam energy: dIS36/dV {be['dIS36_dV']:.3f}, combination {be['d_comb_dV']:.4f} MHz/V")
    th = out["theory_route"]
    print(f"theory mass shifts: 34 {th['mass_shift_34']:.1f}, 36 {th['mass_shift_36']:.1f} MHz;"
          f" 2.5% of the 36 one = {0.025*abs(th['mass_shift_36']):.0f} MHz = {0.025*abs(th['mass_shift_36'])/abs(F):.2f} fm2")
    print()
    print(f"{'run':40s} {'h':>5s} {'unc':>5s} {'count':>5s} {'excess':>6s} {'hits34':>6s} {'K':>5s}")
    for p in adopted:
        print(f"{p['run']:40s} {p['hours']:5.2f} {p['unc']:5.1f} {p['counting']:5.1f} {p['excess']:6.1f}"
              f" {p['hits34']:6d} {p['figure_of_merit']:5.1f}  {p['time_kind']}")
    c = out["combination"]
    print(f"\ncombination {c['stat_MHz']:.2f} MHz from {c['beam_hours']:.1f} h -> K = {c['K_MHz_sqrt_h']:.1f} MHz*sqrt(h);"
          f" counting-only would be {c['counting_only_MHz']:.2f} MHz")
    for k, v in c["hours_for"].items():
        print(f"  sigma34 = {k} MHz needs {v:6.0f} h")
    print("36S in-gate ions needed (k from 34S):", {a: round(b) for a, b in out["ions_36_needed"].items()})
    print("34S sigma*sqrt(hits):", [round(x) for x in out["k_34_count"]])
    print()
    for s in scen:
        print(f"s34 {s['s34']:3.1f} s36 {s['s36']:4.1f} wm {s['wavemeter']:.1f} rho {s['rho']:.0f}:"
              f" C {s['comb_MHz']:5.2f} MHz = {s['comb_fm2']:.4f} fm2;"
              f" dr2(36,32) +-{s['dr2_36_32_unc']:.4f}, dr2(36,34) +-{s['dr2_36_34_unc']:.4f}")


if __name__ == "__main__":
    main()
