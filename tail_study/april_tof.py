"""Are the April RF-series 34S ions inside the [5.1, 5.8] us gate? ToF of all hits, on/off resonance."""
import json, sys
import numpy as np, pandas as pd
scans = json.load(open(r"C:\Users\EMALAB\AppData\Local\Temp\claude\C--Users-EMALAB-Documents-Jackson\dd8a3b9c-2aa0-41ef-82f4-6b455f33f032\scratchpad\survey_cec.json"))
pick = [s for s in scans if s["group"] in ("rf_0428", "rf_0429")] + [s for s in scans if s["file"] in ("34S_3-27-26.csv", "scan_20260508_134603.csv")]
edges = np.arange(3.5, 6.51, 0.1)
for s in pick:
    f = pd.read_csv(s["path"], usecols=["tof", "channel", "bunch_id", "wavemeter_wn1"])
    nb = f["bunch_id"].nunique()
    hits = f[(f["channel"] == 2) & (f["tof"] > 0)]
    tof = hits["tof"].to_numpy() * 1e6
    wn = hits["wavemeter_wn1"].to_numpy()
    # resonance region: the wn third with the most hits in 4.6-6 us
    q = np.quantile(f["wavemeter_wn1"], [0, 1])
    h, _ = np.histogram(tof, edges)
    top = edges[np.argmax(h)]
    print(f"{s['group']} {s['label']} {s['file']:28s} RF {s['cond'].get('RF')}  bunches {nb:6d} hits/bunch {len(tof) / nb:.3f}  "
          f"in[5.1,5.8] {np.sum((tof > 5.1) & (tof < 5.8)):5d}  in[4.25,5.5] {np.sum((tof > 4.25) & (tof < 5.5)):5d}  ToF mode {top:.1f} us  "
          f"hist(3.5-6.5, 0.1us): {' '.join(str(int(v)) for v in h)}")
