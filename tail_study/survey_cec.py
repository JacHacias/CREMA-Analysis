"""Per-scan PicoLog temperatures (all TC-08 channels) for the isotope-shift library scans.

Snapshot of the live LevelDB (never opened live), read with plyvel (system Python 3.12).
Scan start = file-name time (local); duration = bunch-id span / 50 Hz.
"""
import json, os, shutil, subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import plyvel

A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"
D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
MARCH = {  # named analysis file -> DAQ file it is a copy of (identical rows and bunch ids)
    "32S_3-23-26.csv": "final_scan_20260323_190036.csv", "34S_3-23-26.csv": "final_scan_20260323_200439.csv",
    "32S_3-23-26_back.csv": "final_scan_20260323_211659.csv", "34S_3-23-26_back.csv": "final_scan_20260323_214049.csv",
    "32S_3-24-26.csv": "final_scan_20260324_145111.csv", "34S_3-24-26.csv": "final_scan_20260324_153303.csv",
    "32S_3-27-26.csv": "final_scan_20260327_142416.csv", "34S_3-27-26.csv": "final_scan_20260327_150034.csv",
}
MAY = ["scan_20260505_194044.csv", "scan_20260505_194747.csv", "scan_20260506_152634.csv", "scan_20260506_154426.csv",
       "scan_20260508_131731.csv", "scan_20260508_134603.csv", "scan_20260508_140738.csv", "scan_20260508_153504.csv",
       "scan_20260508_155825.csv", "scan_20260511_083642.csv", "scan_20260511_091902.csv", "scan_20260512_144505.csv",
       "scan_20260512_155458.csv", "scan_20260513_102135.csv", "scan_20260513_110715.csv", "scan_20260601_120201.csv",
       "scan_20260601_125135.csv", "scan_20260601_144018.csv", "scan_20260601_151739.csv"]

table = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_scans.json")))
LIVE = os.path.expandvars(r"%LOCALAPPDATA%\PicoLog\Capture-Data")
COPY = os.path.expandvars(r"%TEMP%\picolog_snapshot_survey")
shutil.rmtree(COPY, ignore_errors=True)
subprocess.run(["robocopy", LIVE, COPY, "/R:0", "/W:0", "/NFL", "/NDL", "/NJH", "/NJS"], capture_output=True)
if os.path.exists(os.path.join(COPY, "LOCK")):
    os.remove(os.path.join(COPY, "LOCK"))
t_lo = min(r["t0"] for r in table) - 7200
t_hi = max(r["t1"] for r in table) + 7200
db = plyvel.DB(COPY, create_if_missing=False)
series = {}
for ch in (1, 2, 3, 8):
    prefix = f"tc-08.AO111|179.ch{ch}/1/".encode()
    ts, vs = [], []
    for k, v in db.iterator(prefix=prefix):
        t0 = int(k.split(b"/")[-1]) / 1000.0
        if t0 + 1000 < t_lo or t0 > t_hi:
            continue
        vals = np.frombuffer(v, dtype=">f4").astype(float)
        t = t0 + np.arange(vals.size)
        ok = np.isfinite(vals)
        ts.append(t[ok]); vs.append(vals[ok])
    if ts:
        t, v = np.concatenate(ts), np.concatenate(vs)
        o = np.argsort(t)
        series[ch] = (t[o], v[o])
        print(f"ch{ch}: {t.size} samples {datetime.fromtimestamp(t.min())} .. {datetime.fromtimestamp(t.max())}")
    else:
        print(f"ch{ch}: no samples in range")
db.close()

names = {1: "cec", 2: "T1", 3: "T2", 8: "ionpump"}
for r in table:
    for ch, (t, v) in series.items():
        m = (t >= r["t0"]) & (t <= r["t1"])
        key = names[ch]
        if m.sum() > 10:
            tt, vv = t[m], v[m]
            slope = np.polyfit((tt - tt.mean()) / 3600.0, vv, 1)[0]
            r.update({f"{key}_mean": float(vv.mean()), f"{key}_std": float(vv.std()), f"{key}_min": float(vv.min()),
                      f"{key}_max": float(vv.max()), f"{key}_slope_per_h": float(slope), f"{key}_n": int(m.sum())})
        else:
            r.update({f"{key}_mean": None, f"{key}_n": int(m.sum())})
    print(f"{r['file']:26s} {r['start']} {r['duration_s']/60:5.1f} min  CEC "
          + (f"{r['cec_mean']:6.1f} (min {r['cec_min']:.1f} max {r['cec_max']:.1f}, {r['cec_slope_per_h']:+.1f}/h)" if r.get("cec_mean") else "n/a")
          + "  T1 " + (f"{r['T1_mean']:6.1f}" if r.get("T1_mean") else "n/a")
          + "  T2 " + (f"{r['T2_mean']:6.1f}" if r.get("T2_mean") else "n/a")
          + "  pump " + (f"{r['ionpump_mean']:6.1f}" if r.get("ionpump_mean") else "n/a"))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_cec.json")
json.dump(table, open(out, "w"), indent=1)
# also a 1-min decimated trace around each measurement day for plotting
trace = {}
for ch, (t, v) in series.items():
    bins = (t // 60).astype(np.int64)
    u, inv = np.unique(bins, return_inverse=True)
    trace[names[ch]] = [(float(a * 60 + 30), float(b)) for a, b in zip(u, np.bincount(inv, weights=v) / np.bincount(inv))]
json.dump(trace, open(out.replace(".json", "_trace.json"), "w"))
print("wrote", out)
