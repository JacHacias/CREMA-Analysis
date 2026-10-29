# 36S charge radius via muonic 34S-32S calibration: error budget (2026-10-28)

Script: `error_budget.py` -> `error_budget.json`. Inputs: adopted 34S-32S library (energy-loss-tail
analysis), muonic rms radius differences from Schaller et al., PRC 31, 1007 (1985), calculated
F = -47.9 MHz/fm^2 and K_MS = -327.9 GHz u.

## Observable

With K_MS calibrated on muonic dr2(34,32), the 36S result is

    dr2(36,32) = (dnu36 - r*dnu34)/F + r*dr2_mu(34,32),   r = mu36/mu34 = 1.8904

The optical data fix only C = dnu36 - r*dnu34 (K_MS-free; F multiplies a ~2 MHz residual).

| quantity | muonic | 
|---|---|
| dr2(34,32) | 0.1512(79) fm^2 |
| dr2(36,34) | 0.0856(79) fm^2 |
| dr2(36,32) | 0.2368(111) fm^2 (differences taken as independent) |
| C/F = dr2(36,34) - 0.890 dr2(34,32) | -0.0491(105) fm^2 = +2.35(51) MHz |

Floors that no optical improvement removes: r x 0.0079 = 0.0148 fm^2 on dr2(36,32), or
0.890 x 0.0079 = 0.0070 fm^2 on dr2(36,34) (muonic direct: 0.0079).

## Systematics on C

* Beam energy: dC/dV = 0.025 MHz/V (36-32: 1.762, 34-32: 0.919 MHz/V, cancel to 97%). Negligible;
  higher beam energy is not needed for this.
* Wavemeter (1.8 MHz per shift, Cs estimate): 3.85 MHz on C if uncorrelated between the two
  shifts, 1.6 MHz if fully correlated. Dominant unknown; must be measured for sulfur.
* Theory K_MS route instead (no muonic): 2.5% of the 1139 MHz 36-32 mass shift = 28 MHz = 0.59 fm^2.

## Accumulation time (adopted 34-32 data, 7 groups, 10.6 h beam)

* sigma34 = 4.94 MHz -> K = sigma*sqrt(T) = 16.1 MHz sqrt(h). Counting alone would give 1.79 MHz
  (K = 5.8): 87% of the variance is non-counting (pass-to-pass line motion, ~10 MHz/pass).
* Best runs (05-08 pair 5, 05-12) reach K = 6.6, i.e. near counting-limited.
* Hours for sigma34 = 3 / 2 / 1.5 / 1 / 0.5 MHz: now 29 / 65 / 116 / 260 / 1040 h;
  counting-limited 4 / 8 / 15 / 34 / 135 h.
* 36S: no 36S data from before October 2026 is used (not trustworthy), so there is no measured
  36S rate. Projected from the 34S line (median sigma*sqrt(in-gate ions) = 231 MHz), counting-only:
  sigma36 = 10 / 3 / 1 / 0.6 MHz needs ~530 / 5.9k / 53k / 148k in-gate 36S ions, plus the same
  pass-motion excess as 34S unless that is fixed. Hours follow once the QMF-era 36S rate is known.

## Scenarios (C error -> dr2 errors)

| sigma34 | sigma36 | wavemeter | C (MHz) | dr2(36,34) unc (fm^2) |
|---|---|---|---|---|
| 4.9 | 10 | 1.8, rho 0 | 14.2 | 0.30 |
| 1.5 | 3 | 1.8, rho 0 | 5.6 | 0.12 |
| 1.5 | 3 | 0.2 | 4.2 | 0.087 |
| 1.0 | 1 | 1.8, rho 1 | 2.7 | 0.056 |
| 1.0 | 1 | 0.2 | 2.2 | 0.046 |
| muonic direct | | | | 0.0079 |

Matching the muonic dr2(36,34) needs C to ~0.2 MHz; a meaningful test of the muonic C (2x its
error) needs ~1 MHz, e.g. sigma34 0.35 MHz, sigma36 0.6 MHz, wavemeter 0.2 MHz (uncorrelated).
