# QMF ³⁶S isotope-selective transmission filter (SIMION study)

SIMION/Mathieu study of using the beamline QMF (RFQ, no axial trapping/endcaps) as a
**mass filter to transmit only ³⁶S**, rejecting the far more abundant ³²S (94.99%) and
³⁴S (4.25%). Fixed RF frequency **2.4 MHz** (resonant circuit — cannot be changed);
RF amplitude `V_rf` and DC `V_dc` are the only knobs.

## Result — validated operating point

**V_rf = 1400 V, V_dc = 125 V @ 2.4 MHz** transmits **only ³⁶S**:

| isotope | transmission | where lost |
|---|---|---|
| ³⁶S | **91 %** (182/200) | reaches the 170 mm exit |
| ³⁴S | 0 / 200 | ejected at ~50 mm (filter entrance) |
| ³²S | 0 / 200 | ejected at ~40 mm (filter entrance) |

Robust window: **V_rf ≈ 1380–1430 V, V_dc ≈ 110–150 V**.

The off-mass isotopes are ejected in the first ~10 mm of the filter (deeply radially
unstable) → their suppression is exponential, far below the ~1.5 % bound that 200-ion
counting can resolve — the physical basis for the ~10⁻³–10⁻⁴ rejection ³⁴S's abundance
demands. See `figures/penetration_setpoint.png`.

## Key physics learned
- The **trapping** (timeout) observable ≠ **transmission** (filter) — only transmission matters here.
- Effective field radius r₀ ≈ **rod-center distance (~8.8 mm)**, not the inscribed radius, so the
  Mathieu apex sits at high V_rf (~1.4 kV) rather than the naive ~380 V.
- At nominal V_rf the device is **reject-heavy** (ejects ³⁶S first → isolates ³²S). The
  **reject-light** regime needed for ³⁶S appears only near the real apex (~1.4 kV, feasible on the hardware).

## Layout
- `scripts/` — analysis & plotting (Python) + `QMF.lua` (SIMION workbench program; patched for
  fallback output paths). Key tools:
  - `sweep_vdc_transmission.py` — V_dc sweep at fixed RF, all isotopes (chunked flying, optional pressure).
  - `scan_aq_plane.py` — map transmission/stability over the (a,q) plane for one mass.
  - `scan_vrf_vdc_purity.py` — **2-D (V_rf,V_dc) scan, all 3 isotopes, abundance-weighted ³⁶S purity** (the filter hunt).
  - `diag_penetration.py` — axial penetration-before-loss per isotope (deep-suppression evidence).
  - `plot_*` — Mathieu diagrams, (a,q) maps, RGB purity maps, context/detail panels, writeup figure.
- `drivers/` — PowerShell self-healing run-drivers (launched as Windows Scheduled Tasks; long
  SIMION jobs at 1.4 kV are slow). Paths are machine-specific.
- `figures/` — key deliverables, incl. `36S_filter_writeup_figure.png` (master summary).

> Note: raw scan CSVs and the full plot sets are generated under the SIMION `QMF/data/` tree
> (outside this repo) and are not committed; rerun the scripts to regenerate.
