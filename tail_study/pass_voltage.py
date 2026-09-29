"""Do per-pass line centers follow the per-pass DMM voltage (HV-scale error) or time (drift)?"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = r"C:\Users\EMALAB\Documents\Jackson\CREMA-Analysis"; sys.path.insert(0, REPO)
import matplotlib; matplotlib.use("Agg")
import quick_isotope_shift as qis, rate_spectrum as rs
import pandas as pd

HERE = Path(__file__).resolve().parent
A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"; D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
pp = json.load(open(HERE / "per_pass_tail.json"))
paths = {Path(p).name: p for p in [D + r"\scan_20260508_131731.csv", D + r"\scan_20260508_140738.csv", D + r"\scan_20260508_153504.csv",
                                    D + r"\scan_20260511_083642.csv", D + r"\scan_20260512_144505.csv", D + r"\scan_20260513_102135.csv",
                                    A + r"\32S_3-23-26.csv", A + r"\32S_3-27-26.csv", D + r"\scan_20260601_144018.csv",
                                    D + r"\scan_20260512_155458.csv", D + r"\scan_20260508_134603.csv", A + r"\34S_3-23-26.csv"]}
base = dict(qis.DEFAULT_ANALYSIS_OPTIONS); base.update(voltage_offset_V=184.54201214242858, bootstrap_replicas=0)
allv, allc = [], []
for name, rec in pp.items():
    lab = "34S" if name.startswith("34S") or name in ("scan_20260512_155458.csv", "scan_20260508_134603.csv") else "32S"
    frame = rs.load_scan_frame([Path(paths[name])], ())
    codes, keys = rs._dwell_codes(frame)
    frame["pass"] = keys[codes, 1]
    first = frame.drop_duplicates("bunch_id")
    v = first.groupby("pass")["voltage"].mean().to_numpy() * 5962.49
    vs = first.groupby("pass")["voltage"].std().to_numpy() * 5962.49
    b0 = first.groupby("pass")["bunch_id"].min().to_numpy()
    c = np.array(rec["shape_fixed"]["centers"]); e = np.array(rec["shape_fixed"]["errors"])
    n = min(len(v), len(c))
    v, vs, c, e, b0 = v[:n], vs[:n], c[:n], e[:n], b0[:n]
    ok = np.isfinite(c)
    dc = c - np.nanmean(c); dv = v - np.mean(v)
    slope = 30.84 if lab == "32S" else 29.92
    print(f"{lab} {name:26s} passes {n}: dV(pass) = " + " ".join(f"{x:+6.2f}" for x in dv) + " V (within-pass sd " +
          " ".join(f"{x:.2f}" for x in vs) + ")  dCenter = " + " ".join(f"{x:+6.1f}" for x in dc) +
          f" MHz  -> expected from HV-scale error only if dC ~ k*dV; ratio dC/dV = " +
          " ".join(f"{a / b:+7.1f}" if abs(b) > 0.05 else "   n/a" for a, b in zip(dc, dv)))
    allv += list(dv[ok]); allc += list(dc[ok] / slope)
allv, allc = np.array(allv), np.array(allc)
k = np.polyfit(allv, allc, 1)[0]
print(f"\npooled: center shift (V-equivalent) vs pass DMM offset: slope {k:+.3f} (r = {np.corrcoef(allv, allc)[0, 1]:+.2f}, n={allv.size})")
print(f"spread of per-pass DMM means: {np.std(allv):.3f} V ; spread of per-pass centers: {np.std(allc):.3f} V-equivalent")
