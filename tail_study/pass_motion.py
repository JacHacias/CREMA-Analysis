"""What moves the line from pass to pass? Per-pass covariates vs the frozen-shape per-pass centers
of per_pass_tail.json (shape_fixed): HeNe channel (wavemeter drift), lock residual, DMM voltage,
beam-intensity proxies (off-resonance background, all-mass hits per bunch), in-gate ToF, time.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import rate_spectrum as rs

HERE = Path(__file__).resolve().parent
A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"; D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
PATHS = [D + r"\scan_20260508_131731.csv", D + r"\scan_20260508_140738.csv", D + r"\scan_20260508_153504.csv",
         D + r"\scan_20260511_083642.csv", D + r"\scan_20260512_144505.csv", D + r"\scan_20260513_102135.csv",
         A + r"\32S_3-23-26.csv", A + r"\32S_3-27-26.csv", D + r"\scan_20260601_144018.csv",
         D + r"\scan_20260512_155458.csv", D + r"\scan_20260508_134603.csv", A + r"\34S_3-23-26.csv"]
GATES = {"32S": (4.25, 5.5), "34S": (4.7, 5.8)}
DIVIDER = 5962.49
MHZ_PER_CM = 29979.2458


def per_pass(path, lab):
    frame = rs.load_scan_frame([Path(path)], ())
    codes, keys = rs._dwell_codes(frame)
    frame["pass"] = keys[codes, 1]
    hits = frame["channel"] == 2
    b = frame.drop_duplicates("bunch_id")
    g0, g1 = GATES[lab]
    tof_us = frame["tof"] * 1e6
    in_gate = hits & (tof_us > g0) & (tof_us < g1)
    # bins far from resonance: outer 20% of each pass's bin range, both sides
    nb = frame["scan_bin_index"].max() + 1
    off = (frame["scan_bin_index"] < 0.2 * nb) | (frame["scan_bin_index"] > 0.8 * nb)
    boff = b[(b["scan_bin_index"] < 0.2 * nb) | (b["scan_bin_index"] > 0.8 * nb)]
    rows = []
    for p, bp in b.groupby("pass"):
        fp = frame[frame["pass"] == p]
        n_b = len(bp)
        n_off_b = int((boff["pass"] == p).sum())
        rows.append(dict(
            pass_=int(p), bunches=n_b, t_mid_s=float(bp["bunch_id"].mean()) / 50.0,
            hene=float(bp["wavemeter_wn4"].mean()),
            lock=float((bp["wavemeter_wn1"] - bp["laser_target_wn"]).mean()),
            volt=float(bp["voltage"].mean()) * DIVIDER,
            volt_sd=float(bp["voltage"].std()) * DIVIDER,
            hits_per_bunch=float((hits & (frame["pass"] == p)).sum()) / n_b,
            bg_per_bunch=float((in_gate & off & (frame["pass"] == p)).sum()) / max(n_off_b, 1),
            outgate_per_bunch=float((hits & ~in_gate & (frame["pass"] == p)).sum()) / n_b,
            tof_mean_ns=float(tof_us[in_gate & (frame["pass"] == p)].mean()) * 1e3,
        ))
    return rows


def main():
    pp = json.load(open(HERE / "per_pass_tail.json"))
    table = []
    for path in PATHS:
        name = Path(path).name
        lab = "34S" if name.startswith("34S") or name in ("scan_20260512_155458.csv", "scan_20260508_134603.csv") else "32S"
        rows = per_pass(path, lab)
        c = np.array(pp[name]["shape_fixed"]["centers"]); e = np.array(pp[name]["shape_fixed"]["errors"])
        n = min(len(rows), len(c))
        for r, ci, ei in zip(rows[:n], c[:n], e[:n]):
            r.update(scan=name, iso=lab, center=float(ci), err=float(ei))
            table.append(r)
    json.dump(table, open(HERE / "pass_motion.json", "w"), indent=1)

    keys = ["t_mid_s", "hene", "lock", "volt", "hits_per_bunch", "bg_per_bunch", "outgate_per_bunch", "tof_mean_ns"]
    # within-scan deviations
    dev = {k: [] for k in keys + ["center", "err"]}
    for name in dict.fromkeys(r["scan"] for r in table):
        rs_ = [r for r in table if r["scan"] == name and np.isfinite(r["center"])]
        if len(rs_) < 2:
            continue
        for k in keys + ["center"]:
            v = np.array([r[k] for r in rs_], float)
            dev[k] += list(v - v.mean())
        dev["err"] += [r["err"] for r in rs_]
        print(f"{name:26s} " + " | ".join(
            f"dC {r['center'] - np.mean([q['center'] for q in rs_]):+6.1f} hene {1e4 * (r['hene'] - np.mean([q['hene'] for q in rs_])):+5.2f}"
            f" lock {1e4 * r['lock']:+5.2f} V {r['volt'] - np.mean([q['volt'] for q in rs_]):+5.2f}"
            f" hits/b {r['hits_per_bunch']:.3f} bg {r['bg_per_bunch']:.4f} tof {r['tof_mean_ns']:.0f}"
            for r in rs_))
    dc = np.array(dev["center"])
    print(f"\nn = {dc.size} pass deviations, center sd {dc.std():.1f} MHz, median Fisher err {np.median(dev['err']):.1f} MHz")
    expect = {"hene": -2 * MHZ_PER_CM, "lock": 2 * MHZ_PER_CM, "volt": 30.5}
    for k in keys:
        x = np.array(dev[k])
        if x.std() == 0:
            continue
        r = np.corrcoef(x, dc)[0, 1]
        slope = np.polyfit(x, dc, 1)[0]
        extra = f"; spread x sd {x.std():.3g} -> {abs(expect[k]) * x.std():.1f} MHz if fully coupled" if k in expect else ""
        print(f"  {k:18s} r = {r:+.2f}  slope {slope:+.3g} MHz/unit{extra}")


if __name__ == "__main__":
    main()
