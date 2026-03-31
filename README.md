# CREMA-Analysis

Utilities and notebooks for sulfur isotope analysis, including Doppler-corrected isotope-shift fitting for two and three isotopes.

## Files

- `isotope_shift_analysis.py`
  Two-isotope comparison with Doppler correction, Gaussian peak fitting, panel plots, and propagated fit/HV uncertainties.
- `three_isotope_shift_analysis.py`
  Three-isotope comparison for `32S`, `34S`, and `36S` with separate panels and shifts reported relative to `32S`.
- `Sulfur_plotting.ipynb`
  Notebook workflow for data exploration and analysis.

## Requirements

Install the Python packages used by the scripts:

```python
import numpy as np
import matplotlib.pyplot as plt
import satlas2
```

`satlas2` provides the Gaussian fitting used in the isotope-shift scripts. Install it with `pip install satlas2`.

## Two-Isotope Example

```python
from isotope_shift_analysis import plot_two_isotopes_fit

out = plot_two_isotopes_fit(
    cut_file_1=cut_file_32S,
    cut_file_2=cut_file_34S,
    mass1_u=31.972071,
    mass2_u=33.967867,
    wn_col="wavemeter_wn1",
    bin_width_MHz=20.0,
    tof_gate_us=(4.25, 5.5),
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=1.2,
    geometry="collinear",
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
    bin_width_MHz=20.0,
    tof_gate_us=(4.25, 5.5),
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=1.2,
    geometry="collinear",
)
```

## Notes

- `beam_voltage_unc_V` should include the effect of HV drift/oscillation if the scans were taken at different times.
- You can use either `bins=...` or `bin_width_MHz=...` to control histogram binning. If `bin_width_MHz` is given, it takes precedence.
- You can optionally apply ToF gating inside the analysis functions with `tof_gate_us=(min_us, max_us)`. If you already passed pre-gated cut files, leave this as `None`.
- If your laser geometry is opposite to the ion beam, use `geometry="anticollinear"`.
- The scripts clean the wavemeter column as numeric input before fitting, which helps avoid string-type issues from structured arrays.
