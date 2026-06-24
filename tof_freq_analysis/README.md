# Lineshape / ToF diagnostics (not part of the production pipeline)

Standalone tools from the 2026-06 investigation into the asymmetric 32/34S RIS
lineshapes. **None of these feed the production isotope-shift analysis** — they are
diagnostics kept for the record and for future re-checks. They reuse the pipeline's
own frequency calibration (`isotope_shift_analysis`, `quick_isotope_shift`) and the
library's recorded per-isotope ToF gates, so run them from the repo root.

## Tools (in the repo root)

- **`tof_freq_shear.py`** — builds the 2-D (detuning vs ToF) map for a scan, fits the
  freq–ToF tilt on the resonance core, applies a shear correction, and compares the
  1-D lineshape before/after. Uses each scan's library ToF gate automatically.
  `python tof_freq_shear.py scan_YYYYMMDD_HHMMSS.csv`

- **`lineshape_model_comparison.py`** — refits every 34S-32S library pair with a
  symmetric Voigt vs an asymmetric split-Voigt and compares the resulting isotope-shift
  scatter. `python lineshape_model_comparison.py`

## Reference data

- **`library_tof_gates.csv`** — the per-scan / per-isotope ToF gate each library row
  recorded (`options_json.per_isotope_tof_gates`), with the isotope assigned by matching
  the scan's median wavenumber to `isotope_wavenumber_windows`.

## Findings (summary)

Three analysis-side corrections were tested and **all fail to improve the 34S-32S
isotope shift**:

1. **ToF-frequency shear** (CEC "tilted ellipse" hypothesis): the band is horizontal,
   tilt ≤ 50 MHz/us, ~0% benefit.
2. **Shot (laser-dwell) normalization**: helps the high-statistics 32S lineshape
   (skew −0.68 → −0.20) but *increases* the IS scatter (+129%) because the starved 34S
   scans (hundreds–few thousand ions) get noise-amplified by the division.
3. **Asymmetric split-Voigt**: fits better and the left tail is real/universal
   (wL/wR ≈ 1.5–4.9), but the peak centroid shifts by an isotope- and scan-dependent
   amount that does not cancel in the difference → IS scatter +328%.

The two real levers are at **acquisition** (sweep the laser with equal dwell per
setpoint so the count spectrum is not distorted at the source) and **statistics**
(more 34S counts). The symmetric Voigt remains the right production model because it is
applied identically to both isotopes.
