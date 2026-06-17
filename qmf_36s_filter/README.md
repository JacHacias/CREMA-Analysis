# QMF ³⁶S isotope selectivity — SIMION study

SIMION/Mathieu study of isotope-selecting ³⁶S in the beamline QMF, against the far more
abundant ³²S (94.99%) and ³⁴S (4.25%). Fixed RF frequency **2.4 MHz** (resonant circuit —
cannot be changed); RF amplitude `V_rf` and DC `V_dc` are the only knobs.

Two distinct mechanisms were studied — they select *opposite* ends of the mass range:

| | mechanism | favours | use case |
|---|---|---|---|
| **§1 Transmission filter** | radial Mathieu stability (RFQ, no endcaps) | **light** at nominal V; **³⁶S** near the apex | the operative approach |
| **§2 Ion trapping** | axial RF-pseudopotential confinement | **heavy (³⁶S)** | needs endcaps/axial trap |

---

## §1 — Transmission filter (main result)

Operating the QMF as a **mass filter** (RFQ, **no axial trapping/endcaps**): the observable is
**transmission** (ions reaching the exit), and selectivity comes from radial Mathieu stability.

**Validated operating point: V_rf = 1400 V, V_dc = 125 V @ 2.4 MHz** → transmits **only ³⁶S**:

| isotope | transmission | where lost |
|---|---|---|
| ³⁶S | **91 %** (182/200) | reaches the 170 mm exit |
| ³⁴S | 0 / 200 | ejected at ~50 mm (filter entrance) |
| ³²S | 0 / 200 | ejected at ~40 mm (filter entrance) |

Robust window ≈ **V_rf 1380–1430 V, V_dc 110–150 V**. Off-mass isotopes are ejected in the first
~10 mm of the filter (deeply radially unstable) → exponential suppression, far below the ~1.5 %
bound that 200-ion counting resolves — the basis for the ~10⁻³–10⁻⁴ rejection ³⁴S's abundance needs.

Key physics: effective r₀ ≈ rod-center distance (~8.8 mm), so the Mathieu apex (the reject-light,
³⁶S-selecting regime) sits at ~1.4 kV RF — high but feasible. At nominal V_rf the filter is
*reject-heavy* and isolates ³²S instead.

Figures: `figures/transmission_filter/` — `36S_filter_writeup_figure.png` (master summary),
`context_detail_panel.png`, `context_landscape.png`, `penetration_setpoint.png`.

## §2 — Ion trapping (axial selection)

The original deliverable: **trapped-fraction vs V_dc** (mono + realistic 10 keV energy spread),
showing isotope-selective *axial* trapping (ions that remain confined for the full flight time).

- At fixed RF 448 Vp, scanning V_dc, the axial pseudopotential **traps the heavy ³⁶S at ~28 V**
  (³²S/³⁴S fully rejected), ³²S at ~37.5–39 V; ³⁴S has no clean window (³²S co-traps).
- This favours the **heaviest** isotope — opposite of the transmission filter.
- Confirmed intrinsic (not collisional): pressure-off / zero-emittance runs reproduce the same
  broad, structured ³⁶S window; the breadth is RF-entry-phase averaging, not a removable artifact.
- Abundance-weighted, the 28 V window is a clean pure-³⁶S point — **but only usable with axial
  confinement (endcaps)**, which this device doesn't have, so §1 is the operative route.

Figures: `figures/trapping/` — `vdc_trapping_annotated_compare.png` (two-panel selective regions),
`vdc_trapping_monoenergetic.png`, `vdc_trapping_abundance_weighted.png`,
`vdc_trapping_purity_composition.png`.

`figures/mathieu/` — conceptual (a,q) stability diagrams and the V_dc-scan-as-vertical-lines picture
that connect the two sections.

---

## Layout
- `scripts/` — Python analysis + plotting and the patched `QMF.lua`. Key tools:
  - `sweep_vdc_transmission.py` — V_dc sweep at fixed RF, all isotopes (records both trapping and
    transmission; chunked flying, optional pressure).
  - `scan_aq_plane.py` — map transmission over the (a,q) plane for one mass.
  - `scan_vrf_vdc_purity.py` — **2-D (V_rf,V_dc) scan, all 3 isotopes, abundance-weighted ³⁶S purity** (filter hunt).
  - `diag_penetration.py` — axial penetration-before-loss per isotope (deep-suppression evidence).
  - `plot_*` — Mathieu diagrams, (a,q) / RGB purity maps, abundance/purity, context/detail/writeup panels.
- `drivers/` — PowerShell self-healing run-drivers (launched as Windows Scheduled Tasks; long 1.4 kV
  SIMION jobs are slow). **Paths are machine-specific.**
- `figures/` — key deliverables, grouped by `transmission_filter/`, `trapping/`, `mathieu/`.

> Raw scan CSVs and full plot sets are generated under the SIMION `QMF/data/` tree (outside this
> repo) and are not committed — rerun the scripts to regenerate.
