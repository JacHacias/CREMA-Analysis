# CREMA-Analysis

Utilities and notebooks for sulfur isotope analysis, including Doppler-corrected isotope-shift fitting for two and three isotopes.

## Files

- `isotope_shift_analysis.py`
  Two-isotope comparison with Doppler correction, Voigt peak fitting, panel plots, and propagated fit/HV uncertainties.
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

`satlas2` provides the Voigt fitting used in the isotope-shift scripts. Install it with `pip install satlas2`.

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
