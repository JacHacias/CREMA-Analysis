# Pass-to-pass line motion: what causes it (2026-10-28)

Scripts: `pass_motion.py` (per-pass covariates), `dwell_noise.py` (dwell-level noise + Monte Carlo),
`flank_noise.py` (intensity vs line-jitter split, `flank_noise.png`). Per-pass centers: frozen-shape
fits from `per_pass_tail.json`. Observed scatter: about 12 MHz rms per pass (0.4 V equivalent).

## Ruled out (none is above ~1 MHz)

| candidate | test | result |
|---|---|---|
| wavemeter drift | HeNe channel (wn4) per pass | sd 1.7e-5 cm^-1, about 1 MHz; r = 0.00 |
| HV | DMM per pass | sd 0.04 V, about 1 MHz; r = 0.00 |
| scan direction or wavemeter lag | passes alternate D/U; up-minus-down | +0.5 +- 16.6 MHz, no constant offset |
| lock residual | wn1 - target per pass | alternates with direction, r = -0.13 |
| slow intensity drift skewing the sweep | predicted from the per-scan rate trend | r = -0.06 |
| ToF-frequency shear | earlier study (<= 50 MHz/us) x 16 ns | < 1 MHz |

## What is there

1. **Source-intensity noise at the dwell level.** There is 7-34 % extra multiplicative noise per
   10 s dwell, correlated over a few dwells. In Monte Carlo it gives about 6 MHz rms of
   pass-to-pass scatter, about a quarter of the variance overall. It dominates the two worst
   groups: 06-01 (eta 0.32, MC 17.5 vs observed 18.7 MHz) and 03-27 (eta 0.19).
2. **A real line shift in the May scans.** The excess variance sits on the flanks
   (proportional to (d ln R/dnu)^2), not at the peak: delta = 13-21 MHz for 05-08, 05-11 and 05-13.
   Pooled: eta_I = 0.12, delta = 12.7 MHz.
3. **The shift follows the beam intensity.** Per-pass center versus relative ions per bunch is
   positive in 9 of 11 scans (pooled r = 0.22, about +15 to +50 MHz per doubling, i.e. 0.5-1.6 V).
   More ions give higher energy (collinear: energy gain moves the line up). This matches
   space charge in the RFQ trap raising the extraction energy. The DMM, on the platform HV,
   cannot see it. The evidence is suggestive, not conclusive: only 3-6 passes per scan, and the
   intensity proxy is the resonant signal itself.

## Tests and fixes

* **Direct test:** scan 32S at two trap loads (accumulation time or source power) and look for a
  line shift of tens of MHz per factor 2 in load.
* **Record a load monitor per bunch.** The DAQ keeps only hits in the 4-6 us ToF window. Record
  all masses, or a pickoff after the RFQ, and log the RFQ DC/bias supplies and ablation
  parameters into the scan files. That allows a correction or a gate on load.
* **Reduce or stabilize the load:** run below the space-charge regime, cap the fill, use a steadier
  ablation spot, and use the QMF to remove contaminant ions from the trap.
