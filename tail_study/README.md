# Line-width and energy-loss-tail study (2026-09-29)

Question: do the 32S/34S line widths follow the RFQ settings or the CEC temperature, and
does modeling the asymmetric low-frequency tail give a less uncertain isotope shift?

Model: `rate_spectrum.py` option `tail_model="exponential"` (FFT line shape: a fraction
`f` of the atoms carries an extra exponential energy loss of mean `lambda`), with
`shape_transfer="tail"` (34S takes the 32S tail in volts) and the pair bootstrap
(`pair_bootstrap_shift`). Defaults leave the adopted analysis unchanged.

| step | script | output |
|---|---|---|
| scan list + run-sheet conditions (OneNote archive) | `survey_scans.py` | `survey_scans.json` |
| PicoLog CEC temperatures per scan (system Python, plyvel) | `survey_cec.py` | `survey_cec.json` |
| symmetric + tail fits of 67 scans | `survey_fit.py` | `survey_fit.json`, `.log` |
| correlations (CEC, power, rate, ablation, period) | `survey_corr.py` | `survey_rows.json`, `survey_corr.png` |
| tail side, collinear vs anticollinear | `anticollinear_test.py` | `anticollinear_test.json/.png` |
| within-bunch vs bunch-to-bunch loss (count statistics) | `bunch_f2.py` | `bunch_f2.json/.png` |
| pass-to-pass centers, frozen shape; vs DMM voltage | `per_pass_tail.py`, `pass_voltage.py` | `per_pass_tail.json` |
| beam-energy offset with each line shape | `recalibrate.py` | `recalibrate.json` |
| IS per row, pair bootstrap at each model's offset | `pair_boot_final.py 200 <sym V> <tail V>` | `pair_boot_200_cal.json` |
| combination (9 pairs / 7 run groups) | `combine_final.py` | `is_final_summary.json` |
| summary figure | `summary_fig.py` | `tail_study_summary.png` |

Scripts read and write in this folder; run with `..\.venv\Scripts\python.exe` from here
(`survey_cec.py` needs the system Python with plyvel).

Results (see `tail_study_summary.png`):

* The tail is a kinetic-energy loss inside every bunch: it flips to the high-frequency
  side in anticollinear geometry (10/10 calibration scans), and the per-bunch count
  statistics exclude whole-bunch energy jumps.
* No dependence on the CEC reservoir temperature (206-227 C) within a period. The tail
  fraction changed from ~0.3 (March) to 0.6-0.8 (late April on, after the RFQ work).
  396 nm power broadens only the Lorentzian (11 to 37 MHz HWHM). RFQ settings were
  constant for all IS scans; the April RF-amplitude series has too few 34S ions.
* Pass-to-pass line motion (~13 MHz) is real (survives a frozen line shape) and is not
  seen by the DMM (per-pass voltage spread 0.04 V, no correlation).
* The stored +184.54 V offset came from the legacy raw-count fit. v2 symmetric gives
  +186.54(70) V, v2 tail core +190.77(83) V.
* 34S-32S with each model's own offset, 200-replica pair bootstrap:
  symmetric 579.3(5.8)(0.6) MHz, tail + transfer 580.1(4.9)(0.8) MHz.
