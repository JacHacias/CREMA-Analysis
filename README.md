# CREMA-Analysis

Utilities and notebooks for sulfur isotope analysis, including Doppler-corrected isotope-shift fitting for two and three isotopes.

## Files

- `isotope_shift_analysis.py`
  Two-isotope comparison with Doppler correction, Voigt peak fitting, panel plots, and propagated fit/HV uncertainties.
- `three_isotope_shift_analysis.py`
  Three-isotope comparison for `32S`, `34S`, and `36S` with separate panels and shifts reported relative to `32S`.
- `Sulfur_plotting.ipynb`
  Notebook workflow for data exploration and analysis.
- `dp900_ui/app.py`
  Browser GUI for controlling a Rigol DP932A / DP900-series power supply over LAN socket, VISA, or simulator mode.

## Requirements

Install the Python packages used by the scripts:

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy
```

If `satlas2` is installed, the isotope-shift scripts use it for Voigt fitting.
If it is not installed, they fall back to SciPy's `curve_fit`, which is useful
for quick analysis on this machine without extra setup.

The Rigol DP900 control UI has no required packages for LAN socket or simulator mode. From the repo root, run:

```powershell
python .\dp900_ui\app.py
```

Then open `http://127.0.0.1:8765`.

## Two-Isotope Example

```python
from isotope_shift_analysis import plot_two_isotopes_fit

out = plot_two_isotopes_fit(
    cut_file_1=cut_file_32S,
    cut_file_2=cut_file_34S,
    mass1_u=31.972071,
    mass2_u=33.967867,
    wn_col="wavemeter_wn1",
    frequency_multiplier=2.0,
    bin_width_MHz=20.0,
    tof_gate_us=(4.25, 5.5),
    show_tof_gate_plots=True,
    voltage_col="voltage",
    voltage_multiplier=5962.49,
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=1.2,
    geometry="collinear",
    neutralization="none",
    label1="32S",
    label2="34S",
)
```

## Three-Isotope Example

```python
from three_isotope_shift_analysis import plot_three_isotopes_fit

out = plot_three_isotopes_fit(
    cut_file_32S=cut_file_32S,
    cut_file_34S=cut_file_34S,
    cut_file_36S=cut_file_36S,
    mass32_u=31.972071,
    mass34_u=33.967867,
    mass36_u=35.967081,
    wn_col="wavemeter_wn1",
    frequency_multiplier=2.0,
    bin_width_MHz=20.0,
    tof_gate_us=(4.25, 5.5),
    show_tof_gate_plots=True,
    voltage_col="voltage",
    voltage_multiplier=5962.49,
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=1.2,
    geometry="collinear",
    neutralization="none",
)
```

## Quick Day-End Library Workflow

Use `quick_isotope_shift.py` when you have just taken files and want a fast
centroid/isotope-shift summary saved to a growing data library. The script
infers isotope labels from filenames such as `32S_3-27-26.csv`,
`34S_3-27-26.csv`, and `36S_3-27-26.csv`.

## Spectrum Library GUI

For the lowest-friction end-of-day workflow, launch the local GUI:

```powershell
.\.venv\Scripts\python.exe .\hfs_gui\spectrum_library_gui.py
```

You can also double-click `hfs_gui\launch_spectrum_library_gui.bat`, which frees
port 8766 (stopping any running instance) and then starts the GUI.

Then open `http://127.0.0.1:8766` if the browser does not open
automatically. Paste filenames or full paths into the file box, enter the
collection date/time, and click `Analyze and Add`. The GUI writes the same
library files as the command-line workflow and shows recent fit plots. It also
regenerates library-wide stability plots from the saved rows:

- total `32S` / `34S` centroid stability
- `34S-32S` isotope-shift stability
- `36S-32S` isotope-shift stability when `36S` rows exist

The local library has been seeded from the existing sulfur files in
`..\S data for analysis`, including the March 23, March 24, and March 27
main and `_back` background spectra. Use `Rebuild From Folder` in the GUI to
replace the library from that folder again. The rebuild form has an
`Include _back background files` checkbox so you can choose whether those scans
are included.

By default the GUI and command-line workflow automatically remove bad scan
passes before fitting. A bad scan pass is inferred from resets in
`scan_bin_index` and rejected if it has poor bin coverage, too few events,
invalid wavemeter data, a collapsed wavemeter range, or an extreme
`spectrum_peak` outlier. Each library row records `scans_removed`,
`points_removed`, and a JSON `bad_scan_filter` audit trail.

If a file name does not contain the isotope label, such as raw DAQ names like
`scan_20260506_140648.csv`, enter labels in the `Isotope labels` box in the
same order as the pasted files, for example `32S,34S`.

You can paste multiple consecutive files for the same isotope. The workflow
bundles all files with the same isotope label before bad-scan filtering, ToF
gating, and fitting. For raw DAQ filenames, repeat labels in the same order as
the files, for example `32S,32S,34S,34S`.

Raw DAQ filenames are also checked against the expected wavemeter region for
this sulfur transition. If a file labeled `34S` has a median wavemeter value in
the `32S` region, the GUI refuses the fit and reports the suspicious file
instead of saving a bad library row.

The GUI has separate `32S`, `34S`, and `36S` ToF gate fields. Fill the isotope
gates that apply to the files being analyzed; the workflow pre-gates each
isotope independently before fitting.

The spectrum library and GUI tables report centroids, isotope shifts, and
uncertainties in MHz. The fitting functions still keep GHz internally and
convert at the library/plot display boundary.

`Rebuild From Folder` leaves `36S` unchecked by default, so the March 27 `36S`
scan is not included unless you explicitly enable `Include 36S files when
rebuilding`.

Two-isotope run:

```powershell
python .\quick_isotope_shift.py `
  --data-dir "..\S data for analysis" `
  --collection-date 2026-03-27 `
  --collection-time "afternoon" `
  --run-label "sulfur_2026-03-27" `
  --transition "12625 cm-1 line" `
  32S_3-27-26.csv 34S_3-27-26.csv
```

Three-isotope run:

```powershell
python .\quick_isotope_shift.py `
  --data-dir "..\S data for analysis" `
  --collection-date 2026-03-27 `
  --collection-time "afternoon" `
  --run-label "sulfur_2026-03-27" `
  32S_3-27-26.csv 34S_3-27-26.csv 36S_3-27-26.csv
```

The quick workflow writes:

- `analysis_plots/`: PNG fit plots for the run.
- `data_library/isotope_shift_library.csv`: spreadsheet-friendly summary rows.
- `data_library/isotope_shift_library.jsonl`: one JSON record per result row for
  later scripting/publication organization.

Analysis defaults live in `analysis_defaults.json`. You can either edit that
file or override settings from the command line, for example:

```powershell
python .\quick_isotope_shift.py `
  --config .\analysis_defaults.json `
  --data-dir "..\S data for analysis" `
  --tof-gate-us 4.25,5.5 `
  --beam-voltage-unc-V 1.2 `
  32S_3-27-26.csv 34S_3-27-26.csv
```

## Notes

- By default, the Doppler correction uses the per-event voltage column when available: `voltage_col="voltage"` scaled by `voltage_multiplier=5962.49`, the `B_HVD2` high-voltage divider factor. If the column is missing, the scripts fall back to `beam_voltage_V`.
- Set `use_voltage_column=False` to force the older fixed-voltage behavior.
- `beam_voltage_unc_V` is applied as an additional voltage offset uncertainty on top of either the voltage-column values or the fixed fallback value.
- Pass neutral isotope masses as `mass*_u`; the Doppler correction internally subtracts `charge_e` electron masses to use the ion mass for the accelerated 1+ beam.
- The neutralizer correction is explicit and opt-in through `neutralization`.
  Use `neutralization="none"` for the historical charged-ion beta correction, `neutralization="electron_capture"` to conserve the incoming sulfur momentum while changing from the ion mass to the neutral atom mass after electron pickup, or `neutralization="sodium_charge_exchange"` to use a simple collinear two-body sulfur/sodium charge-exchange kinematic model.
- `neutralization="electron_capture"` follows the kinetic-energy framework `T_atom = -m_atom c^2 + sqrt((m_atom c^2)^2 + T_ion^2 + 2 T_ion m_ion c^2)`, then computes neutral sulfur beta from the neutral total energy. No sodium/electron lab-velocity term is included.
- For `neutralization="sodium_charge_exchange"`, `sodium_collision_branch="forward"` selects the forward charge-exchange root closest to the incoming sulfur velocity. `sodium_collision_branch="momentum_transfer"` selects the large momentum-transfer root and should be treated as a bounding/stress-test model unless you have evidence for hard sodium scattering.
- The sodium neutralizer mass defaults to `SODIUM_MASS_U = 22.9897692820`; the sodium target is treated as stationary in the lab-frame charge-exchange estimate.
- `frequency_multiplier=2.0` is appropriate when the wavemeter column records the fundamental laser before optical doubling. Set it to `1.0` if the column is already the doubled spectroscopy frequency.
- You can use either `bins=...` or `bin_width_MHz=...` to control histogram binning. If `bin_width_MHz` is given, it takes precedence.
- You can optionally apply ToF gating inside the analysis functions with `tof_gate_us=(min_us, max_us)`. If you already passed pre-gated cut files, leave this as `None`.
- Set `show_tof_gate_plots=True` to display raw and gated ToF histograms before the isotope-shift fit.
- If your laser geometry is opposite to the ion beam, use `geometry="anticollinear"`.
- The scripts clean the wavemeter column as numeric input before fitting, which helps avoid string-type issues from structured arrays.
