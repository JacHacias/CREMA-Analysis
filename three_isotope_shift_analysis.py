import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import wofz

from plot_style import apply_publication_style, style_axes

try:
    import satlas2
except ModuleNotFoundError:
    satlas2 = None


C = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
ELECTRON_MASS_U = 5.48579909065e-4
B_HVD2 = 5962.49
SODIUM_MASS_U = 22.9897692820
DEFAULT_HENE_WAVELENGTH_NM = 632.992


def voigt(x, amplitude, center, sigma_g, gamma_l, background):
    sigma_g = max(float(sigma_g), 1e-12)
    gamma_l = max(float(gamma_l), 1e-12)
    z = ((np.asarray(x, dtype=float) - center) + 1j * gamma_l) / (sigma_g * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma_g * np.sqrt(2.0 * np.pi))
    peak = np.max(profile)
    if not np.isfinite(peak) or peak <= 0:
        return np.full_like(np.asarray(x, dtype=float), background, dtype=float)
    return amplitude * (profile / peak) + background


def _voigt_fwhm(sigma_g, gamma_l):
    gauss_fwhm = 2.354820045 * max(float(sigma_g), 1e-12)
    lorentz_fwhm = 2.0 * max(float(gamma_l), 1e-12)
    return 0.5346 * lorentz_fwhm + np.sqrt(0.2166 * lorentz_fwhm**2 + gauss_fwhm**2)


def _fallback_center_uncertainty(x_fit, y_fit, sigma_g, gamma_l, background):
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    if x_fit.size > 1:
        dx = float(np.median(np.diff(x_fit)))
    else:
        dx = 1e-3

    signal = np.clip(y_fit - float(background), 0.0, None)
    n_eff = float(np.sum(signal))
    if n_eff <= 1.0:
        n_eff = max(float(np.sum(y_fit > background)), 1.0)

    fwhm = _voigt_fwhm(sigma_g, gamma_l)
    return max(fwhm / (2.355 * np.sqrt(n_eff)), 0.5 * dx)


if satlas2 is not None:
    class VoigtModel(satlas2.Model):
        def __init__(self, amplitude, center, sigma_g, gamma_l, background, name="Voigt", prefunc=None):
            super().__init__(name, prefunc=prefunc)
            self.params = {
                "amplitude": satlas2.Parameter(value=amplitude, min=0.0, max=np.inf, vary=True),
                "center": satlas2.Parameter(value=center, vary=True),
                "sigma_g": satlas2.Parameter(value=sigma_g, min=1e-12, max=np.inf, vary=True),
                "gamma_l": satlas2.Parameter(value=gamma_l, min=1e-12, max=np.inf, vary=True),
                "background": satlas2.Parameter(value=background, vary=True),
            }

        def f(self, x):
            x = self.transform(x)
            amplitude = self.params["amplitude"].value
            center = self.params["center"].value
            sigma_g = self.params["sigma_g"].value
            gamma_l = self.params["gamma_l"].value
            background = self.params["background"].value
            return voigt(x, amplitude, center, sigma_g, gamma_l, background)


def _sulfur_velocity_after_sodium_collision(
    velocity_m_s,
    mass_u,
    charge_e=1,
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
):
    ion_mass_kg = (float(mass_u) - float(charge_e) * ELECTRON_MASS_U) * AMU
    sulfur_neutral_mass_kg = float(mass_u) * AMU
    sodium_neutral_mass_kg = float(sodium_mass_u) * AMU
    sodium_ion_mass_kg = (float(sodium_mass_u) - ELECTRON_MASS_U) * AMU

    velocity_m_s = np.asarray(velocity_m_s, dtype=float)

    total_momentum = ion_mass_kg * velocity_m_s
    total_kinetic = 0.5 * ion_mass_kg * velocity_m_s**2

    # Nonrelativistic two-body kinematics are more than sufficient at 10 kV
    # sulfur beam speeds. The two roots correspond to forward charge exchange
    # and the large momentum-transfer branch.
    a = sulfur_neutral_mass_kg + sulfur_neutral_mass_kg**2 / sodium_ion_mass_kg
    b = -2.0 * total_momentum * sulfur_neutral_mass_kg / sodium_ion_mass_kg
    c = total_momentum**2 / sodium_ion_mass_kg - 2.0 * total_kinetic
    discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)

    root_plus = (-b + np.sqrt(discriminant)) / (2.0 * a)
    root_minus = (-b - np.sqrt(discriminant)) / (2.0 * a)

    branch = sodium_collision_branch.lower()
    if branch in ("forward", "spectator", "charge_exchange"):
        return np.where(
            np.abs(root_plus - velocity_m_s) <= np.abs(root_minus - velocity_m_s),
            root_plus,
            root_minus,
        )
    if branch in ("momentum_transfer", "hard_collision", "scattered"):
        return np.where(
            np.abs(root_plus - velocity_m_s) > np.abs(root_minus - velocity_m_s),
            root_plus,
            root_minus,
        )
    raise ValueError(
        "sodium_collision_branch must be 'forward' or 'momentum_transfer'."
    )


def beam_beta_after_cec(
    mass_u,
    beam_voltage_V=10000.0,
    charge_e=1,
    neutralization="none",
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
):
    ion_mass_u = float(mass_u) - float(charge_e) * ELECTRON_MASS_U
    if ion_mass_u <= 0:
        raise ValueError("Ion mass must be positive after electron-mass correction.")

    m = ion_mass_u * AMU
    q = charge_e * E_CHARGE
    KE = q * beam_voltage_V

    gamma = 1.0 + KE / (m * C**2)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    mode = str(neutralization).lower()
    if mode in ("none", "ion", "charged"):
        return beta

    if mode in ("electron_capture", "neutral", "neutral_mass"):
        # Momentum-conserving capture, written in the kinetic-energy form:
        # T_atom = -m_atom c^2 + sqrt((m_atom c^2)^2 + T_ion^2 + 2 T_ion m_ion c^2)
        atom_mass_energy = float(mass_u) * AMU * C**2
        atom_total_energy = np.sqrt(
            atom_mass_energy**2 + KE**2 + 2.0 * KE * m * C**2
        )
        return np.sqrt(1.0 - (atom_mass_energy / atom_total_energy) ** 2)

    if mode in ("sodium_charge_exchange", "sodium_collision", "charge_exchange"):
        ion_velocity = beta * C
        sulfur_velocity = _sulfur_velocity_after_sodium_collision(
            ion_velocity,
            mass_u,
            charge_e=charge_e,
            sodium_mass_u=sodium_mass_u,
            sodium_collision_branch=sodium_collision_branch,
        )
        return sulfur_velocity / C

    raise ValueError(
        "neutralization must be 'none', 'electron_capture', or 'sodium_charge_exchange'."
    )


def doppler_correct_ghz(
    nu_lab_ghz,
    mass_u,
    beam_voltage_V=10000.0,
    charge_e=1,
    geometry="collinear",
    *,
    neutralization="none",
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
):
    beta = beam_beta_after_cec(
        mass_u,
        beam_voltage_V=beam_voltage_V,
        charge_e=charge_e,
        neutralization=neutralization,
        sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    if geometry == "collinear":
        factor = gamma * (1.0 - beta)
    elif geometry == "anticollinear":
        factor = gamma * (1.0 + beta)
    else:
        raise ValueError("geometry must be 'collinear' or 'anticollinear'")

    return np.asarray(nu_lab_ghz, dtype=float) * factor


def fit_histogram_peak(x, counts, smooth_sigma_bins=2):
    counts = np.asarray(counts, dtype=float)
    x = np.asarray(x, dtype=float)

    smooth_counts = gaussian_filter1d(counts, smooth_sigma_bins)
    i_max = np.argmax(smooth_counts)

    peak_height = smooth_counts[i_max]
    half_max = 0.5 * peak_height

    i_left = i_max
    while i_left > 0 and smooth_counts[i_left] > half_max:
        i_left -= 1

    i_right = i_max
    while i_right < len(smooth_counts) - 1 and smooth_counts[i_right] > half_max:
        i_right += 1

    fwhm_bins = max(i_right - i_left, 3)
    fit_half_width_bins = max(4, int(1.5 * fwhm_bins))

    i0 = max(0, i_max - fit_half_width_bins)
    i1 = min(len(x), i_max + fit_half_width_bins + 1)

    x_fit = x[i0:i1]
    y_fit = counts[i0:i1]

    x0_guess = x[i_max]
    edge_points = np.r_[y_fit[:2], y_fit[-2:]]
    y0_guess = max(np.median(edge_points), 0.0)
    A_guess = max(np.max(y_fit) - y0_guess, 1.0)
    if i_right > i_left:
        sigma_guess = max((x[i_right] - x[i_left]) / 2.355, 1e-3)
    else:
        sigma_guess = max((x_fit.max() - x_fit.min()) / 6, 1e-3)
    gamma_guess = max(0.5 * sigma_guess, 1e-3)
    yerr = np.sqrt(y_fit)
    yerr[yerr <= 0] = 1.0

    if len(x_fit) > 1:
        dx = abs(x_fit[1] - x_fit[0])
    else:
        dx = 1e-3
    width_max = max((x_fit.max() - x_fit.min()) / 3.0, 1e-2)

    def _run_fit(fix_gamma=False, fix_background=False):
        if satlas2 is None:
            lower = [0.0, x_fit.min(), max(dx / 2.0, 1e-3), max(dx / 2.0, 1e-3), 0.0]
            upper = [np.inf, x_fit.max(), width_max, width_max, max(np.max(y_fit), 1.0)]
            p0 = [A_guess, x0_guess, sigma_guess, gamma_guess, y0_guess]
            p0 = [
                min(max(value, low), high) if np.isfinite(high) else max(value, low)
                for value, low, high in zip(p0, lower, upper)
            ]

            if fix_gamma and fix_background:
                def model(x, amplitude, center, sigma_g):
                    return voigt(x, amplitude, center, sigma_g, gamma_guess, y0_guess)

                popt_small, pcov = curve_fit(
                    model,
                    x_fit,
                    y_fit,
                    p0=[p0[0], p0[1], p0[2]],
                    sigma=yerr,
                    absolute_sigma=True,
                    bounds=([lower[0], lower[1], lower[2]], [upper[0], upper[1], upper[2]]),
                    maxfev=20000,
                )
                popt = np.array([popt_small[0], popt_small[1], popt_small[2], gamma_guess, y0_guess], dtype=float)
                perr_small = np.sqrt(np.diag(pcov))
                perr = np.array([perr_small[0], perr_small[1], perr_small[2], 0.0, 0.0], dtype=float)
                return popt, perr

            if fix_gamma:
                def model(x, amplitude, center, sigma_g, background):
                    return voigt(x, amplitude, center, sigma_g, gamma_guess, background)

                popt_small, pcov = curve_fit(
                    model,
                    x_fit,
                    y_fit,
                    p0=[p0[0], p0[1], p0[2], p0[4]],
                    sigma=yerr,
                    absolute_sigma=True,
                    bounds=([lower[0], lower[1], lower[2], lower[4]], [upper[0], upper[1], upper[2], upper[4]]),
                    maxfev=20000,
                )
                popt = np.array([popt_small[0], popt_small[1], popt_small[2], gamma_guess, popt_small[3]], dtype=float)
                perr_small = np.sqrt(np.diag(pcov))
                perr = np.array([perr_small[0], perr_small[1], perr_small[2], 0.0, perr_small[3]], dtype=float)
                return popt, perr

            popt, pcov = curve_fit(
                voigt,
                x_fit,
                y_fit,
                p0=p0,
                sigma=yerr,
                absolute_sigma=True,
                bounds=(lower, upper),
                maxfev=20000,
            )
            return popt, np.sqrt(np.diag(pcov))

        model = VoigtModel(A_guess, x0_guess, sigma_guess, gamma_guess, y0_guess)
        model.params["center"].min = x_fit.min()
        model.params["center"].max = x_fit.max()
        model.params["sigma_g"].min = max(dx / 2.0, 1e-3)
        model.params["sigma_g"].max = width_max
        model.params["gamma_l"].min = max(dx / 2.0, 1e-3)
        model.params["gamma_l"].max = width_max
        model.params["background"].min = 0.0
        model.params["background"].max = max(np.max(y_fit), 1.0)
        if fix_gamma:
            model.params["gamma_l"].vary = False
        if fix_background:
            model.params["background"].vary = False

        source = satlas2.Source(x_fit, y_fit, yerr=yerr, name="HistogramData")
        source.addModel(model)

        fitter = satlas2.Fitter()
        fitter.addSource(source)
        fitter.fit(method="leastsq")

        params = model.params
        popt = np.array(
            [
                params["amplitude"].value,
                params["center"].value,
                params["sigma_g"].value,
                params["gamma_l"].value,
                params["background"].value,
            ],
            dtype=float,
        )
        perr = np.array(
            [
                getattr(params["amplitude"], "unc", np.nan),
                getattr(params["center"], "unc", np.nan),
                getattr(params["sigma_g"], "unc", np.nan),
                getattr(params["gamma_l"], "unc", np.nan),
                getattr(params["background"], "unc", np.nan),
            ],
            dtype=float,
        )
        return popt, perr

    popt, perr = _run_fit()
    if not np.isfinite(popt[1]) or not np.isfinite(perr[1]):
        popt_retry, perr_retry = _run_fit(fix_gamma=True)
        if np.isfinite(popt_retry[1]):
            popt, perr = popt_retry, perr_retry
    if not np.isfinite(popt[1]) or not np.isfinite(perr[1]):
        popt_retry, perr_retry = _run_fit(fix_gamma=True, fix_background=True)
        if np.isfinite(popt_retry[1]):
            popt, perr = popt_retry, perr_retry
    if not np.isfinite(perr[1]):
        perr[1] = _fallback_center_uncertainty(x_fit, y_fit, popt[2], popt[3], popt[4])

    return popt, None, perr, x_fit


def clean_numeric_column(dat, col):
    x = np.array(dat[col], dtype=float)
    return x[np.isfinite(x)]


def wn_to_lab_ghz(wn_cm, frequency_multiplier=2.0):
    return np.asarray(wn_cm, dtype=float) * float(frequency_multiplier) * C * 100.0 * 1e-9


def air_refractive_index_peck_reeder(wavelength_air_nm):
    """Index of standard air for visible wavelengths using Peck-Reeder."""
    wavelength_air_um = np.asarray(wavelength_air_nm, dtype=float) * 1e-3
    sigma2 = (1.0 / wavelength_air_um) ** 2
    n_minus_1 = 1e-8 * (
        8060.51
        + 2480990.0 / (132.274 - sigma2)
        + 17455.7 / (39.32957 - sigma2)
    )
    return 1.0 + n_minus_1


def vacuum_to_air_wavelength_nm(wavelength_vacuum_nm, iterations=6):
    """Convert a vacuum wavelength to the corresponding standard-air wavelength."""
    wavelength_air = np.asarray(wavelength_vacuum_nm, dtype=float)
    for _ in range(iterations):
        wavelength_air = np.asarray(wavelength_vacuum_nm, dtype=float) / air_refractive_index_peck_reeder(wavelength_air)
    return wavelength_air


def hene_reference_wavenumber_cm(
    hene_reference_wavelength_nm=DEFAULT_HENE_WAVELENGTH_NM,
    hene_reference_wavelength_medium="vacuum",
    hene_wavenumber_medium="vacuum",
):
    """
    Build the HeNe reference wavenumber in the same convention as wavemeter_wn4.

    The 632.992 nm HeNe wavelength is converted to its standard-air wavelength
    internally. The sulfur data files here report wavemeter_wn4 near 15798 cm^-1,
    so the default comparison medium is the vacuum-equivalent wavenumber.
    """
    wavelength_nm = float(hene_reference_wavelength_nm)
    wavelength_medium = str(hene_reference_wavelength_medium).lower()
    if wavelength_medium == "vacuum":
        wavelength_vacuum_nm = wavelength_nm
        wavelength_air_nm = float(vacuum_to_air_wavelength_nm(wavelength_vacuum_nm))
    elif wavelength_medium == "air":
        wavelength_air_nm = wavelength_nm
        wavelength_vacuum_nm = wavelength_air_nm * float(air_refractive_index_peck_reeder(wavelength_air_nm))
    else:
        raise ValueError("hene_reference_wavelength_medium must be 'vacuum' or 'air'.")

    wavenumber_medium = str(hene_wavenumber_medium).lower()
    if wavenumber_medium == "vacuum":
        return 1e7 / wavelength_vacuum_nm
    if wavenumber_medium == "air":
        return 1e7 / wavelength_air_nm
    raise ValueError("hene_wavenumber_medium must be 'vacuum' or 'air'.")


def hene_correct_wavenumber(
    dat,
    wn_col,
    hene_col="wavemeter_wn4",
    hene_reference_wn=None,
    hene_reference_wavelength_nm=DEFAULT_HENE_WAVELENGTH_NM,
    hene_reference_wavelength_medium="vacuum",
    hene_wavenumber_medium="vacuum",
):
    wn_cm = np.array(dat[wn_col], dtype=float)
    if (
        hene_col is None
        or getattr(dat, "dtype", None) is None
        or dat.dtype.names is None
        or hene_col not in dat.dtype.names
    ):
        return wn_cm

    hene_wn = np.array(dat[hene_col], dtype=float)
    finite = np.isfinite(hene_wn)
    if not np.any(finite):
        return wn_cm

    if hene_reference_wn is not None:
        reference = float(hene_reference_wn)
    elif hene_reference_wavelength_nm is not None:
        reference = hene_reference_wavenumber_cm(
            hene_reference_wavelength_nm=hene_reference_wavelength_nm,
            hene_reference_wavelength_medium=hene_reference_wavelength_medium,
            hene_wavenumber_medium=hene_wavenumber_medium,
        )
    else:
        reference = float(np.nanmedian(hene_wn[finite]))
    return wn_cm - (hene_wn - reference)


def _lab_frequency_and_voltage(
    dat,
    wn_col,
    frequency_multiplier=2.0,
    beam_voltage_V=10000.0,
    voltage_col="voltage",
    voltage_multiplier=B_HVD2,
    use_voltage_column=True,
    voltage_offset_V=0.0,
    use_hene_calibration=False,
    hene_col="wavemeter_wn4",
    hene_reference_wn=None,
    hene_reference_wavelength_nm=DEFAULT_HENE_WAVELENGTH_NM,
    hene_reference_wavelength_medium="vacuum",
    hene_wavenumber_medium="vacuum",
):
    if use_hene_calibration:
        wn_cm = hene_correct_wavenumber(
            dat,
            wn_col,
            hene_col=hene_col,
            hene_reference_wn=hene_reference_wn,
            hene_reference_wavelength_nm=hene_reference_wavelength_nm,
            hene_reference_wavelength_medium=hene_reference_wavelength_medium,
            hene_wavenumber_medium=hene_wavenumber_medium,
        )
    else:
        wn_cm = np.array(dat[wn_col], dtype=float)
    has_voltage = (
        use_voltage_column
        and getattr(dat, "dtype", None) is not None
        and dat.dtype.names is not None
        and voltage_col in dat.dtype.names
    )

    if has_voltage:
        voltage_V = np.array(dat[voltage_col], dtype=float) * float(voltage_multiplier)
        mask = np.isfinite(wn_cm) & np.isfinite(voltage_V)
        voltage_V = voltage_V[mask] + float(voltage_offset_V)
        voltage_source = "column"
    else:
        mask = np.isfinite(wn_cm)
        voltage_V = np.full(np.count_nonzero(mask), float(beam_voltage_V) + float(voltage_offset_V), dtype=float)
        voltage_source = "fixed"

    nu_lab = wn_to_lab_ghz(wn_cm[mask], frequency_multiplier=frequency_multiplier)
    return nu_lab, voltage_V, voltage_source


def apply_tof_gate(dat, tof_gate_us=None, tof_col="tof"):
    if tof_gate_us is None:
        return dat

    if tof_col not in dat.dtype.names:
        raise KeyError(f"Column '{tof_col}' not found. Available: {dat.dtype.names}")

    if len(tof_gate_us) != 2:
        raise ValueError("tof_gate_us must be a (min_us, max_us) pair.")

    tmin_us, tmax_us = [float(v) for v in tof_gate_us]
    if tmax_us <= tmin_us:
        raise ValueError("tof_gate_us must satisfy max_us > min_us.")

    t_us = np.array(dat[tof_col], dtype=float) * 1e6
    mask = np.isfinite(t_us) & (t_us > tmin_us) & (t_us < tmax_us)
    gated = dat[mask]

    if gated.size == 0:
        raise ValueError(
            f"No events remain after ToF gate {tof_gate_us} us on column '{tof_col}'."
        )

    return gated


def plot_tof_gate_summary(dat, tof_gate_us=None, tof_col="tof", bins=100, label="Data"):
    apply_publication_style()
    if tof_col not in dat.dtype.names:
        raise KeyError(f"Column '{tof_col}' not found. Available: {dat.dtype.names}")

    t_us = np.array(dat[tof_col], dtype=float) * 1e6
    raw_t_us = t_us[np.isfinite(t_us) & (t_us > 0)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].hist(raw_t_us, bins=bins, color="C0", alpha=0.8)
    axes[0].set_title(f"{label} Raw ToF", fontweight="bold")
    axes[0].set_xlabel("ToF (us)", fontweight="bold")
    axes[0].set_ylabel("Counts", fontweight="bold")

    if tof_gate_us is None:
        gated_t_us = raw_t_us
        gate_title = f"{label} ToF (no gate)"
    else:
        tmin_us, tmax_us = [float(v) for v in tof_gate_us]
        gate_mask = np.isfinite(t_us) & (t_us > tmin_us) & (t_us < tmax_us)
        gated_t_us = t_us[gate_mask]
        gate_title = f"{label} Gated ToF ({tmin_us:.3f} to {tmax_us:.3f} us)"

    axes[1].hist(gated_t_us, bins=bins, color="C1", alpha=0.8)
    axes[1].set_title(gate_title, fontweight="bold")
    axes[1].set_xlabel("ToF (us)", fontweight="bold")
    axes[1].set_ylabel("Counts", fontweight="bold")

    for ax in axes:
        style_axes(ax)

    plt.tight_layout()
    plt.show()


def _resolve_histogram_bins(x, bins=120, bin_width_MHz=None):
    if bin_width_MHz is None:
        return bins

    # Keep bin_width_MHz in spectroscopy-frequency units even when the
    # wavemeter axis is scaled by optical frequency doubling.
    bin_width_GHz = float(bin_width_MHz) / 1000.0
    if bin_width_GHz <= 0:
        raise ValueError("bin_width_MHz must be positive.")

    x = np.asarray(x, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Histogram data must be finite.")

    if x_max <= x_min:
        return np.array([x_min - 0.5 * bin_width_GHz, x_max + 0.5 * bin_width_GHz], dtype=float)

    start = x_min - 0.5 * bin_width_GHz
    stop = x_max + 1.5 * bin_width_GHz
    edges = np.arange(start, stop, bin_width_GHz, dtype=float)

    if edges.size < 2:
        edges = np.array([x_min - 0.5 * bin_width_GHz, x_max + 0.5 * bin_width_GHz], dtype=float)

    return edges


def _occupied_xlim(centers, counts, fallback_x, include_points=None):
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    fallback_x = np.asarray(fallback_x, dtype=float)
    include_points = [] if include_points is None else [float(v) for v in include_points]

    occupied = counts > 0
    if np.any(occupied):
        x_min = float(np.min(centers[occupied]))
        x_max = float(np.max(centers[occupied]))
        if centers.size > 1:
            dx = float(np.median(np.diff(centers)))
        else:
            dx = max(abs(x_max - x_min), 1e-3)
    else:
        x_min = float(np.min(fallback_x))
        x_max = float(np.max(fallback_x))
        dx = max(abs(x_max - x_min) / 20.0, 1e-3)

    if fallback_x.size:
        x_min = min(x_min, float(np.min(fallback_x)))
        x_max = max(x_max, float(np.max(fallback_x)))

    if include_points:
        x_min = min([x_min] + include_points)
        x_max = max([x_max] + include_points)

    x_min -= 0.5 * dx
    x_max += 0.5 * dx

    if x_max <= x_min:
        pad = max(dx, 1e-3)
    else:
        pad = max(2.0 * dx, 0.08 * (x_max - x_min))

    return x_min - pad, x_max + pad


def _poisson_yerr(counts):
    counts = np.asarray(counts, dtype=float)
    return np.sqrt(np.clip(counts, 1.0, None))


def _estimate_peak_snr(fit_params):
    amplitude = max(float(fit_params[0]), 0.0)
    background = max(float(fit_params[4]), 0.0)
    peak_total = amplitude + background
    noise = np.sqrt(max(peak_total, 1.0))
    return amplitude / noise


def fit_quality_metrics(centers, counts, fit_params, x_fit_window):
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    model_counts = voigt(centers, *fit_params)
    yerr = np.sqrt(np.clip(counts, 1.0, None))
    x_fit_window = np.asarray(x_fit_window, dtype=float)
    if x_fit_window.size:
        in_fit = (centers >= np.min(x_fit_window)) & (centers <= np.max(x_fit_window))
    else:
        in_fit = np.ones_like(centers, dtype=bool)
    dof = max(int(np.sum(in_fit)) - len(fit_params), 1)
    reduced_chi2 = float(np.sum(((counts[in_fit] - model_counts[in_fit]) / yerr[in_fit]) ** 2) / dof)
    peak_index = int(np.argmax(counts)) if counts.size else 0
    peak_model = float(max(model_counts[peak_index], 1e-12)) if counts.size else np.nan
    return {
        "peak_to_model_max": float(counts[peak_index] / peak_model) if counts.size else np.nan,
        "reduced_chi2": reduced_chi2,
        "max_bin_center_MHz": float(centers[peak_index] * 1000.0) if counts.size else np.nan,
        "fit_window_min_MHz": float(np.min(x_fit_window) * 1000.0) if x_fit_window.size else np.nan,
        "fit_window_max_MHz": float(np.max(x_fit_window) * 1000.0) if x_fit_window.size else np.nan,
    }


def _fit_center_from_voltage(
    dat,
    mass_u,
    beam_voltage_V,
    wn_col,
    bins,
    charge_e,
    geometry,
    nu0_ref,
    bin_width_MHz=None,
    frequency_multiplier=2.0,
    voltage_col="voltage",
    voltage_multiplier=B_HVD2,
    use_voltage_column=True,
    voltage_offset_V=0.0,
    use_hene_calibration=False,
    hene_col="wavemeter_wn4",
    hene_reference_wn=None,
    hene_reference_wavelength_nm=DEFAULT_HENE_WAVELENGTH_NM,
    hene_reference_wavelength_medium="vacuum",
    hene_wavenumber_medium="vacuum",
    neutralization="none",
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
):
    nu_lab, voltage_V, _ = _lab_frequency_and_voltage(
        dat,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        voltage_offset_V=voltage_offset_V,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col,
        hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
    )
    nu = doppler_correct_ghz(
        nu_lab,
        mass_u,
        voltage_V,
        charge_e,
        geometry,
        neutralization=neutralization,
        sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    x = nu - nu0_ref

    hist_bins = _resolve_histogram_bins(x, bins=bins, bin_width_MHz=bin_width_MHz)
    counts, edges = np.histogram(x, bins=hist_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p, cov, err, x_fit = fit_histogram_peak(centers, counts)
    return {
        "center": float(p[1]),
        "center_fit_unc": float(err[1]),
        "x": x,
        "counts": counts,
        "centers": centers,
        "fit_params": p,
        "x_fit_window": x_fit,
        "fit_quality": fit_quality_metrics(centers, counts, p, x_fit),
    }


def plot_three_isotopes_fit(
    cut_file_32S,
    cut_file_34S,
    cut_file_36S,
    mass32_u=31.972071,
    mass34_u=33.967867,
    mass36_u=35.967081,
    wn_col="wavemeter_wn1",
    bins=120,
    bin_width_MHz=None,
    frequency_multiplier=2.0,
    tof_gate_us=None,
    tof_col="tof",
    show_tof_gate_plots=False,
    tof_plot_bins=100,
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=0.0,
    voltage_col="voltage",
    voltage_multiplier=B_HVD2,
    voltage_offset_V=0.0,
    use_voltage_column=True,
    use_hene_calibration=False,
    hene_col="wavemeter_wn4",
    hene_reference_wn=None,
    hene_reference_wavelength_nm=DEFAULT_HENE_WAVELENGTH_NM,
    hene_reference_wavelength_medium="vacuum",
    hene_wavenumber_medium="vacuum",
    charge_e=1,
    geometry="collinear",
    neutralization="none",
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
):
    apply_publication_style()
    if show_tof_gate_plots:
        plot_tof_gate_summary(cut_file_32S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="32S")
        plot_tof_gate_summary(cut_file_34S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="34S")
        plot_tof_gate_summary(cut_file_36S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="36S")

    cut_file_32S = apply_tof_gate(cut_file_32S, tof_gate_us=tof_gate_us, tof_col=tof_col)
    cut_file_34S = apply_tof_gate(cut_file_34S, tof_gate_us=tof_gate_us, tof_col=tof_col)
    cut_file_36S = apply_tof_gate(cut_file_36S, tof_gate_us=tof_gate_us, tof_col=tof_col)

    nu32_lab, voltage32_V, voltage32_source = _lab_frequency_and_voltage(
        cut_file_32S,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        voltage_offset_V=voltage_offset_V,
        use_voltage_column=use_voltage_column,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col,
        hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
    )
    nu34_lab, voltage34_V, voltage34_source = _lab_frequency_and_voltage(
        cut_file_34S,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        voltage_offset_V=voltage_offset_V,
        use_voltage_column=use_voltage_column,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col,
        hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
    )
    nu36_lab, voltage36_V, voltage36_source = _lab_frequency_and_voltage(
        cut_file_36S,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        voltage_offset_V=voltage_offset_V,
        use_voltage_column=use_voltage_column,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col,
        hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
    )

    nu32 = doppler_correct_ghz(
        nu32_lab, mass32_u, voltage32_V, charge_e, geometry,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    nu34 = doppler_correct_ghz(
        nu34_lab, mass34_u, voltage34_V, charge_e, geometry,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    nu36 = doppler_correct_ghz(
        nu36_lab, mass36_u, voltage36_V, charge_e, geometry,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )

    nu0 = np.median(nu32)

    res32 = _fit_center_from_voltage(
        cut_file_32S, mass32_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
        voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        voltage_offset_V=voltage_offset_V,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col, hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    res34 = _fit_center_from_voltage(
        cut_file_34S, mass34_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
        voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        voltage_offset_V=voltage_offset_V,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col, hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    res36 = _fit_center_from_voltage(
        cut_file_36S, mass36_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
        voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        voltage_offset_V=voltage_offset_V,
        use_hene_calibration=use_hene_calibration,
        hene_col=hene_col, hene_reference_wn=hene_reference_wn,
        hene_reference_wavelength_nm=hene_reference_wavelength_nm,
        hene_reference_wavelength_medium=hene_reference_wavelength_medium,
        hene_wavenumber_medium=hene_wavenumber_medium,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )

    for res in (res32, res34, res36):
        res["center_voltage_unc"] = 0.0

    if beam_voltage_unc_V > 0:
        for dat, mass_u, res in [
            (cut_file_32S, mass32_u, res32),
            (cut_file_34S, mass34_u, res34),
            (cut_file_36S, mass36_u, res36),
        ]:
            c_plus = _fit_center_from_voltage(
                dat, mass_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
                bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
                voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
                use_voltage_column=use_voltage_column, voltage_offset_V=voltage_offset_V + beam_voltage_unc_V,
                use_hene_calibration=use_hene_calibration,
                hene_col=hene_col, hene_reference_wn=hene_reference_wn,
                hene_reference_wavelength_nm=hene_reference_wavelength_nm,
                hene_reference_wavelength_medium=hene_reference_wavelength_medium,
                hene_wavenumber_medium=hene_wavenumber_medium,
                neutralization=neutralization, sodium_mass_u=sodium_mass_u,
                sodium_collision_branch=sodium_collision_branch,
            )["center"]
            c_minus = _fit_center_from_voltage(
                dat, mass_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
                bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
                voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
                use_voltage_column=use_voltage_column, voltage_offset_V=voltage_offset_V - beam_voltage_unc_V,
                use_hene_calibration=use_hene_calibration,
                hene_col=hene_col, hene_reference_wn=hene_reference_wn,
                hene_reference_wavelength_nm=hene_reference_wavelength_nm,
                hene_reference_wavelength_medium=hene_reference_wavelength_medium,
                hene_wavenumber_medium=hene_wavenumber_medium,
                neutralization=neutralization, sodium_mass_u=sodium_mass_u,
                sodium_collision_branch=sodium_collision_branch,
            )["center"]
            res["center_voltage_unc"] = abs(c_plus - c_minus) / 2.0

    for res in (res32, res34, res36):
        res["center_total_unc"] = np.sqrt(res["center_fit_unc"]**2 + res["center_voltage_unc"]**2)

    shift_34_32 = res34["center"] - res32["center"]
    shift_36_32 = res36["center"] - res32["center"]

    shift_34_32_fit_unc = np.sqrt(res34["center_fit_unc"]**2 + res32["center_fit_unc"]**2)
    shift_36_32_fit_unc = np.sqrt(res36["center_fit_unc"]**2 + res32["center_fit_unc"]**2)

    shift_34_32_voltage_unc = np.sqrt(res34["center_voltage_unc"]**2 + res32["center_voltage_unc"]**2)
    shift_36_32_voltage_unc = np.sqrt(res36["center_voltage_unc"]**2 + res32["center_voltage_unc"]**2)

    shift_34_32_total_unc = np.sqrt(shift_34_32_fit_unc**2 + shift_34_32_voltage_unc**2)
    shift_36_32_total_unc = np.sqrt(shift_36_32_fit_unc**2 + shift_36_32_voltage_unc**2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    display_scale = 1000.0
    display_unit = "MHz"
    plot_info = [
        (axes[0], res32, "32S", "C0"),
        (axes[1], res34, "34S", "C1"),
        (axes[2], res36, "36S", "C2"),
    ]
    xlims = [
        _occupied_xlim(res32["centers"], res32["counts"], res32["x"], include_points=[res32["center"], 0.0]),
        _occupied_xlim(res34["centers"], res34["counts"], res34["x"], include_points=[res34["center"], 0.0]),
        _occupied_xlim(res36["centers"], res36["counts"], res36["x"], include_points=[res36["center"], 0.0]),
    ]
    x_left = min(limit[0] for limit in xlims)
    x_right = max(limit[1] for limit in xlims)

    for ax, res, label, color in plot_info:
        xfit = np.linspace(res["centers"].min(), res["centers"].max(), 2000)
        yfit = voigt(xfit, *res["fit_params"])

        ax.errorbar(
            res["centers"] * display_scale,
            res["counts"],
            yerr=_poisson_yerr(res["counts"]),
            fmt="o",
            ms=4.5,
            color=color,
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            label=label,
        )
        ax.plot(xfit * display_scale, yfit, color=color, lw=2)
        ax.axvline(
            res["center"] * display_scale,
            color=color,
            linestyle="--",
            label=f"center = {res['center'] * display_scale:.1f} +/- {res['center_total_unc'] * display_scale:.1f} {display_unit}",
        )
        ax.axvline(0.0, color="k", linestyle=":", label=r"$\nu_0$")
        ax.set_ylabel("Counts", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        style_axes(ax)
        ax.legend()

    axes[-1].set_xlabel(r"Corrected frequency relative to $\nu_0$ (MHz)", fontweight="bold")
    axes[-1].set_xlim(x_left * display_scale, x_right * display_scale)
    fig.suptitle("Doppler-Corrected Sulfur Isotope Comparison", fontweight="bold", fontsize="x-large")
    plt.tight_layout()
    plt.show()

    print(f"32S center: {res32['center'] * display_scale:.3f} +/- {res32['center_total_unc'] * display_scale:.3f} MHz")
    print(f"  estimated peak SNR: {_estimate_peak_snr(res32['fit_params']):.2f}")
    print(f"34S center: {res34['center'] * display_scale:.3f} +/- {res34['center_total_unc'] * display_scale:.3f} MHz")
    print(f"  estimated peak SNR: {_estimate_peak_snr(res34['fit_params']):.2f}")
    print(f"36S center: {res36['center'] * display_scale:.3f} +/- {res36['center_total_unc'] * display_scale:.3f} MHz")
    print(f"  estimated peak SNR: {_estimate_peak_snr(res36['fit_params']):.2f}")
    print()
    print(f"Isotope shift (34S - 32S): {shift_34_32 * display_scale:.3f} +/- {shift_34_32_total_unc * display_scale:.3f} MHz")
    print(f"  fit contribution: {shift_34_32_fit_unc * display_scale:.3f} MHz")
    print(f"  voltage contribution: {shift_34_32_voltage_unc * display_scale:.3f} MHz")
    print()
    print(f"Isotope shift (36S - 32S): {shift_36_32 * display_scale:.3f} +/- {shift_36_32_total_unc * display_scale:.3f} MHz")
    print(f"  fit contribution: {shift_36_32_fit_unc * display_scale:.3f} MHz")
    print(f"  voltage contribution: {shift_36_32_voltage_unc * display_scale:.3f} MHz")
    print()
    print(f"nu0 reference: {nu0 * display_scale:.3f} MHz")
    print(f"32S voltage source: {voltage32_source} (mean {float(np.mean(voltage32_V)):.3f} V)")
    print(f"34S voltage source: {voltage34_source} (mean {float(np.mean(voltage34_V)):.3f} V)")
    print(f"36S voltage source: {voltage36_source} (mean {float(np.mean(voltage36_V)):.3f} V)")
    print(f"Neutralization model: {neutralization}")
    if str(neutralization).lower() in ("sodium_charge_exchange", "sodium_collision", "charge_exchange"):
        print(f"  sodium collision branch: {sodium_collision_branch}")

    return {
        "nu0_GHz": float(nu0),
        "32S": res32,
        "34S": res34,
        "36S": res36,
        "shift_34_32_GHz": float(shift_34_32),
        "shift_34_32_total_unc_GHz": float(shift_34_32_total_unc),
        "shift_36_32_GHz": float(shift_36_32),
        "shift_36_32_total_unc_GHz": float(shift_36_32_total_unc),
        "tof_gate_us": tof_gate_us,
        "tof_col": tof_col,
        "show_tof_gate_plots": bool(show_tof_gate_plots),
        "frequency_multiplier": float(frequency_multiplier),
        "voltage_col": voltage_col,
        "voltage_multiplier": float(voltage_multiplier),
        "voltage_offset_V": float(voltage_offset_V),
        "use_voltage_column": bool(use_voltage_column),
        "use_hene_calibration": bool(use_hene_calibration),
        "hene_col": hene_col,
        "hene_reference_wn": hene_reference_wn,
        "hene_reference_wavelength_nm": hene_reference_wavelength_nm,
        "hene_reference_wavelength_medium": hene_reference_wavelength_medium,
        "hene_wavenumber_medium": hene_wavenumber_medium,
        "voltage32_source": voltage32_source,
        "voltage34_source": voltage34_source,
        "voltage36_source": voltage36_source,
        "voltage32_mean_V": float(np.mean(voltage32_V)),
        "voltage34_mean_V": float(np.mean(voltage34_V)),
        "voltage36_mean_V": float(np.mean(voltage36_V)),
        "neutralization": neutralization,
        "sodium_mass_u": float(sodium_mass_u),
        "sodium_collision_branch": sodium_collision_branch,
        "num_points_32S": int(cut_file_32S.size),
        "num_points_34S": int(cut_file_34S.size),
        "num_points_36S": int(cut_file_36S.size),
    }


if __name__ == "__main__":
    print("Import this module in your notebook and call plot_three_isotopes_fit(...).")
