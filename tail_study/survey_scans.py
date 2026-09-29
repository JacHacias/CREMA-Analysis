"""Scan list for the width/shape survey: library scans + April RF series + May-15 396-power series.

Conditions come from the OneNote run sheets (MIT Beamline > Sulfur, archive of 2026-07-16):
RF = RF-amplifier input amplitude (V; 5 V = 330 Vpp), P396 = 396 nm power at the CREMA
entrance (mW; None = not recorded), NRI = non-resonant ionization laser, ABL = ablation
flash-lamp voltage, iris (mm). Run-sheet times are scan END times (checked against file
mtimes); CEC is taken from PicoLog separately (sheet values are snapshots).
"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

A = r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis"
D = r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data"
U = r"C:\Users\EMALAB\data"
MARCH_DAQ = {
    "32S_3-23-26.csv": "20260323_190036", "34S_3-23-26.csv": "20260323_200439",
    "32S_3-23-26_back.csv": "20260323_211659", "34S_3-23-26_back.csv": "20260323_214049",
    "32S_3-24-26.csv": "20260324_145111", "34S_3-24-26.csv": "20260324_153303",
    "32S_3-27-26.csv": "20260327_142416", "34S_3-27-26.csv": "20260327_150034",
}
G32_OLD, G34_OLD = (4.25, 5.5), (5.1, 5.8)
G32_MAY, G34_MAY = (4.25, 5.25), (4.7, 5.4)
# (label, file, gate, group, conditions)
C_0323 = dict(RF=5, P396=4.5, NRI="532nm 1000V", ABL=895, iris=33.5)
C_0324 = dict(RF=5, P396=16.5, NRI="532nm 1050V", ABL=895, iris=34)
C_0327 = dict(RF=5, P396=16.5, NRI="532nm 1100V", ABL=895, iris=32, chD_width_ms=1.0)
C_MAY_FULL = dict(RF=5, P396=None, NRI="1064nm 820V", ABL=915, iris=32)
SCANS = [
    ("32S", A + r"\32S_3-23-26.csv", G32_OLD, "library", C_0323),
    ("34S", A + r"\34S_3-23-26.csv", G34_OLD, "library", C_0323),
    ("32S", A + r"\32S_3-23-26_back.csv", G32_OLD, "library", C_0323),
    ("34S", A + r"\34S_3-23-26_back.csv", G34_OLD, "library", C_0323),
    ("32S", A + r"\32S_3-24-26.csv", G32_OLD, "library", C_0324),
    ("34S", A + r"\34S_3-24-26.csv", G34_OLD, "library", C_0324),
    ("32S", A + r"\32S_3-27-26.csv", G32_OLD, "library", C_0327),
    ("34S", A + r"\34S_3-27-26.csv", G34_OLD, "library", C_0327),
    ("32S", D + r"\scan_20260505_194044.csv", G32_OLD, "library", dict(C_MAY_FULL, NRI="1064nm 840V", ABL=897)),
    ("34S", D + r"\scan_20260505_194747.csv", G34_OLD, "library", dict(C_MAY_FULL, NRI="1064nm 840V", ABL=897)),
    ("32S", D + r"\scan_20260506_152634.csv", G32_OLD, "library", C_MAY_FULL),
    ("34S", D + r"\scan_20260506_154426.csv", G34_OLD, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260508_131731.csv", G32_OLD, "library", C_MAY_FULL),
    ("34S", D + r"\scan_20260508_134603.csv", G34_OLD, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260508_140738.csv", G32_OLD, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260508_153504.csv", G32_OLD, "library", C_MAY_FULL),
    ("34S", D + r"\scan_20260508_155825.csv", G34_OLD, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260511_083642.csv", G32_MAY, "library", dict(C_MAY_FULL, P396=1.2, polarizer=150)),
    ("32S", D + r"\scan_20260512_144505.csv", G32_MAY, "library", dict(C_MAY_FULL, P396=1.25, polarizer=165)),
    ("34S", D + r"\scan_20260512_155458.csv", G34_MAY, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260513_102135.csv", G32_MAY, "library", dict(C_MAY_FULL, P396=1.25, polarizer=165)),
    ("34S", D + r"\scan_20260513_110715.csv", G34_MAY, "library", C_MAY_FULL),
    ("32S", D + r"\scan_20260601_120201.csv", G32_OLD, "library", dict(C_MAY_FULL, P396=1.16, ABL=895, iris=30)),
    ("34S", D + r"\scan_20260601_125135.csv", G34_MAY, "library", dict(C_MAY_FULL, P396=2.3, ABL=895, iris=30)),
    ("32S", D + r"\scan_20260601_144018.csv", G32_OLD, "library", dict(C_MAY_FULL, P396=1.2, ABL=895, iris=30)),
    ("34S", D + r"\scan_20260601_151739.csv", (4.8, 5.6), "library", dict(C_MAY_FULL, P396=2.3, ABL=895, iris=30)),
]
# 2026-04-28 RF-amplitude series (34S unless noted), 532 nm NRI 1100 V, ABL 897.
C_APR = dict(P396=None, NRI="532nm 1100V", ABL=897)
for stamp, lab, rf, iris in [("152426", "34S", 10, 32), ("155635", "34S", 10, 32), ("160800", "34S", 10, 32),
                             ("163342", "32S", 5, 32), ("170424", "32S", 5, 20), ("171621", "34S", 5, 25),
                             ("172400", "34S", 5, 32), ("172922", "34S", 5, 29), ("173503", "34S", 10, 29),
                             ("174045", "34S", 10, 29), ("175915", "34S", 5, 29), ("181722", "34S", 11, 29),
                             ("182316", "34S", 13, 29), ("182850", "34S", 14, 29), ("183411", "34S", 15, 29),
                             ("183927", "34S", 16, 29), ("184430", "34S", 16.5, 29), ("184937", "34S", 17, 29)]:
    SCANS.append((lab, D + rf"\scan_20260428_{stamp}.csv", G32_OLD if lab == "32S" else G34_OLD, "rf_0428",
                  dict(C_APR, RF=rf, iris=iris)))
for stamp, lab, rf in [("133305", "34S", 15), ("140210", "32S", 5), ("144021", "34S", 15), ("151812", "32S", 5),
                       ("155145", "34S", 15), ("161553", "34S", 15), ("163445", "34S", 15), ("171205", "32S", 5),
                       ("181321", "34S", 5), ("182329", "34S", 5), ("184336", "34S", 5)]:
    SCANS.append((lab, D + rf"\scan_20260429_{stamp}.csv", G32_OLD if lab == "32S" else G34_OLD, "rf_0429",
                  dict(C_APR, RF=rf, iris=29)))
# 2026-05-15 396 nm power-broadening series (32S), 1064 nm NRI 820 V, ABL 899, CEC ~215.
C_PB = dict(RF=5, NRI="1064nm 820V", ABL=899, iris=32)
for stamp, mw, folder in [("120957", 7.2, U), ("114136", 2.2, U), ("133004", 1.2, U), ("134807", 0.145, U),
                          ("140614", 0.095, U), ("143012", 0.035, U), ("145139", 0.01, U), ("151416", 6.7, U),
                          ("153202", 7.1, U), ("162054", 5.5, D), ("165934", 4.9, D), ("172054", 8.7, D)]:
    SCANS.append(("32S", folder + rf"\scan_20260515_{stamp}.csv", G32_MAY, "power_0515", dict(C_PB, P396=mw)))

out = []
for lab, path, gate, group, cond in SCANS:
    name = Path(path).name
    stamp = MARCH_DAQ.get(name) or name.replace("scan_", "").replace(".csv", "")
    start = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    b = pd.read_csv(path, usecols=["bunch_id"])["bunch_id"].to_numpy()
    dur = float(b.max() - b.min()) / 50.0
    out.append(dict(label=lab, file=name, path=path, gate=list(gate), group=group, cond=cond,
                    start=start.isoformat(), duration_s=dur, t0=start.timestamp(), t1=start.timestamp() + dur))
    print(f"{group:10s} {lab} {name:28s} {start}  {dur / 60:5.1f} min  {cond}")
here = Path(__file__).resolve().parent
json.dump(out, open(here / "survey_scans.json", "w"), indent=1)
print(len(out), "scans")
