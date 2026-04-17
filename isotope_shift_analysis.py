import numpy as np
import matplotlib.pyplot as plt
import satlas2
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz

from plot_style import apply_publication_style, style_axes


C = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
ELECTRON_MASS_U = 5.48579909065e-4
B_HVD2 = 5962.49
SODIUM_MASS_U = 22.9897692820


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
    kinetic_energy = q * beam_voltage_V

    gamma = 1.0 + kinetic_energy / (m * C**2)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    mode = str(neutralization).lower()
    if mode in ("none", "ion", "charged"):
        return beta

    if mode in ("electron_capture", "neutral", "neutral_mass"):
        # Momentum-conserving capture, written in the kinetic-energy form:
        # T_atom = -m_atom c^2 + sqrt((m_atom c^2)^2 + T_ion^2 + 2 T_ion m_ion c^2)
        atom_mass_energy = float(mass_u) * AMU * C**2
        atom_total_energy = np.sqrt(
            atom_mass_energy**2 + kinetic_energy**2 + 2.0 * kinetic_energy * m * C**2
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

    geometry = geometry.lower()
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

    center_guess = x[i_max]
    edge_points = np.r_[y_fit[:2], y_fit[-2:]]
    background_guess = max(np.median(edge_points), 0.0)
    amplitude_guess = max(np.max(y_fit) - background_guess, 1.0)
    if i_right > i_left:
        sigma_guess = max((x[i_right] - x[i_left]) / 2.355, 1e-3)
    else:
        sigma_guess = max((x_fit.max() - x_fit.min()) / 6.0, 1e-3)
    gamma_guess = max(0.5 * sigma_guess, 1e-3)
    yerr = np.sqrt(y_fit)
    yerr[yerr <= 0] = 1.0

    if len(x_fit) > 1:
        dx = abs(x_fit[1] - x_fit[0])
    else:
        dx = 1e-3
    width_max = max((x_fit.max() - x_fit.min()) / 3.0, 1e-2)

    def _run_fit(fix_gamma=False, fix_background=False):
        model = VoigtModel(amplitude_guess, center_guess, sigma_guess, gamma_guess, background_guess)
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


def _lab_frequency_and_voltage(
    dat,
    wn_col,
    frequency_multiplier=2.0,
    beam_voltage_V=10000.0,
    voltage_col="voltage",
    voltage_multiplier=B_HVD2,
    use_voltage_column=True,
    voltage_offset_V=0.0,
):
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
    )
    nu_rest = doppler_correct_ghz(
        nu_lab,
        mass_u,
        voltage_V,
        charge_e,
        geometry,
        neutralization=neutralization,
        sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    x = nu_rest - nu0_ref

    hist_bins = _resolve_histogram_bins(x, bins=bins, bin_width_MHz=bin_width_MHz)
    counts, edges = np.histogram(x, bins=hist_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    popt, pcov, perr, x_fit = fit_histogram_peak(centers, counts)
    return popt[1], perr[1], x, counts, centers, popt, x_fit


def plot_two_isotopes_fit(
    cut_file_1,
    cut_file_2,
    mass1_u,
    mass2_u,
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
    use_voltage_column=True,
    charge_e=1,
    geometry="collinear",
    neutralization="none",
    sodium_mass_u=SODIUM_MASS_U,
    sodium_collision_branch="forward",
    label1="32S",
    label2="34S",
):
    """
    Compare two gated isotope spectra with Doppler correction and simple Voigt fits.

    Returns fit centers and uncertainties for each isotope and the propagated isotope shift.
    """
    apply_publication_style()
    if show_tof_gate_plots:
        plot_tof_gate_summary(cut_file_1, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label=label1)
        plot_tof_gate_summary(cut_file_2, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label=label2)

    cut_file_1 = apply_tof_gate(cut_file_1, tof_gate_us=tof_gate_us, tof_col=tof_col)
    cut_file_2 = apply_tof_gate(cut_file_2, tof_gate_us=tof_gate_us, tof_col=tof_col)

    nu1_lab, voltage1_V, voltage1_source = _lab_frequency_and_voltage(
        cut_file_1,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
    )
    nu2_lab, voltage2_V, voltage2_source = _lab_frequency_and_voltage(
        cut_file_2,
        wn_col,
        frequency_multiplier=frequency_multiplier,
        beam_voltage_V=beam_voltage_V,
        voltage_col=voltage_col,
        voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
    )

    nu1 = doppler_correct_ghz(
        nu1_lab,
        mass1_u,
        voltage1_V,
        charge_e,
        geometry,
        neutralization=neutralization,
        sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    nu2 = doppler_correct_ghz(
        nu2_lab,
        mass2_u,
        voltage2_V,
        charge_e,
        geometry,
        neutralization=neutralization,
        sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )

    nu0 = 0.5 * (np.median(nu1) + np.median(nu2))

    center1, dcenter1_fit, x1, counts1, centers1, p1, xfit_window1 = _fit_center_from_voltage(
        cut_file_1, mass1_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
        voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )
    center2, dcenter2_fit, x2, counts2, centers2, p2, xfit_window2 = _fit_center_from_voltage(
        cut_file_2, mass2_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
        voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
        use_voltage_column=use_voltage_column,
        neutralization=neutralization, sodium_mass_u=sodium_mass_u,
        sodium_collision_branch=sodium_collision_branch,
    )

    isotope_shift = float(center2) - float(center1)
    isotope_shift_fit_unc = np.sqrt(dcenter1_fit**2 + dcenter2_fit**2)

    dcenter1_V = 0.0
    dcenter2_V = 0.0
    isotope_shift_V_unc = 0.0

    if beam_voltage_unc_V > 0:
        c1_plus, _, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_1, mass1_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
            bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
            voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
            use_voltage_column=use_voltage_column, voltage_offset_V=beam_voltage_unc_V,
            neutralization=neutralization, sodium_mass_u=sodium_mass_u,
            sodium_collision_branch=sodium_collision_branch,
        )
        c1_minus, _, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_1, mass1_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
            bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
            voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
            use_voltage_column=use_voltage_column, voltage_offset_V=-beam_voltage_unc_V,
            neutralization=neutralization, sodium_mass_u=sodium_mass_u,
            sodium_collision_branch=sodium_collision_branch,
        )
        dcenter1_V = abs(float(c1_plus) - float(c1_minus)) / 2.0

        c2_plus, _, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_2, mass2_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
            bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
            voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
            use_voltage_column=use_voltage_column, voltage_offset_V=beam_voltage_unc_V,
            neutralization=neutralization, sodium_mass_u=sodium_mass_u,
            sodium_collision_branch=sodium_collision_branch,
        )
        c2_minus, _, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_2, mass2_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
            bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier,
            voltage_col=voltage_col, voltage_multiplier=voltage_multiplier,
            use_voltage_column=use_voltage_column, voltage_offset_V=-beam_voltage_unc_V,
            neutralization=neutralization, sodium_mass_u=sodium_mass_u,
            sodium_collision_branch=sodium_collision_branch,
        )
        dcenter2_V = abs(float(c2_plus) - float(c2_minus)) / 2.0

        shift_plus = float(c2_plus) - float(c1_plus)
        shift_minus = float(c2_minus) - float(c1_minus)
        isotope_shift_V_unc = abs(shift_plus - shift_minus) / 2.0

    dcenter1_total = np.sqrt(dcenter1_fit**2 + dcenter1_V**2)
    dcenter2_total = np.sqrt(dcenter2_fit**2 + dcenter2_V**2)
    isotope_shift_total_unc = np.sqrt(isotope_shift_fit_unc**2 + isotope_shift_V_unc**2)

    xfit1 = np.linspace(centers1.min(), centers1.max(), 2000)
    xfit2 = np.linspace(centers2.min(), centers2.max(), 2000)
    yfit1 = voigt(xfit1, *p1)
    yfit2 = voigt(xfit2, *p2)
    xlim1 = _occupied_xlim(centers1, counts1, x1, include_points=[center1, 0.0])
    xlim2 = _occupied_xlim(centers2, counts2, x2, include_points=[center2, 0.0])
    x_left = min(xlim1[0], xlim2[0])
    x_right = max(xlim1[1], xlim2[1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].errorbar(
        centers1,
        counts1,
        yerr=_poisson_yerr(counts1),
        fmt="o",
        ms=4.5,
        color="C0",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        label=label1,
    )
    axes[0].plot(xfit1, yfit1, color="C0", lw=2)
    axes[0].axvline(
        center1,
        color="C0",
        linestyle="--",
        label=f"center = {center1:.3f} +/- {dcenter1_total:.3f} GHz",
    )
    axes[0].axvline(0.0, color="k", linestyle=":", label=r"$\nu_0$")
    axes[0].set_ylabel("Counts", fontweight="bold")
    axes[0].set_title(label1, fontweight="bold")
    style_axes(axes[0])
    axes[0].legend()

    axes[1].errorbar(
        centers2,
        counts2,
        yerr=_poisson_yerr(counts2),
        fmt="o",
        ms=4.5,
        color="C1",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        label=label2,
    )
    axes[1].plot(xfit2, yfit2, color="C1", lw=2)
    axes[1].axvline(
        center2,
        color="C1",
        linestyle="--",
        label=f"center = {center2:.3f} +/- {dcenter2_total:.3f} GHz",
    )
    axes[1].axvline(0.0, color="k", linestyle=":", label=r"$\nu_0$")
    axes[1].set_ylabel("Counts", fontweight="bold")
    axes[1].set_xlabel(r"Corrected frequency relative to $\nu_0$ (GHz)", fontweight="bold")
    axes[1].set_title(label2, fontweight="bold")
    style_axes(axes[1])
    axes[1].legend()
    axes[1].set_xlim(x_left, x_right)

    fig.suptitle("Doppler-Corrected Isotope Comparison", fontweight="bold", fontsize="x-large")
    plt.tight_layout()
    plt.show()

    print(f"{label1} center: {center1:.6f} +/- {dcenter1_total:.6f} GHz")
    print(f"  fit contribution: {dcenter1_fit:.6f} GHz")
    print(f"  voltage contribution: {dcenter1_V:.6f} GHz")
    print(f"  estimated peak SNR: {_estimate_peak_snr(p1):.2f}")
    print(f"{label2} center: {center2:.6f} +/- {dcenter2_total:.6f} GHz")
    print(f"  fit contribution: {dcenter2_fit:.6f} GHz")
    print(f"  voltage contribution: {dcenter2_V:.6f} GHz")
    print(f"  estimated peak SNR: {_estimate_peak_snr(p2):.2f}")
    print(f"Isotope shift ({label2} - {label1}): {isotope_shift:.6f} +/- {isotope_shift_total_unc:.6f} GHz")
    print(f"  fit contribution: {isotope_shift_fit_unc:.6f} GHz")
    print(f"  voltage contribution: {isotope_shift_V_unc:.6f} GHz")
    print(f"nu0 reference: {nu0:.6f} GHz")
    print(f"{label1} voltage source: {voltage1_source} (mean {float(np.mean(voltage1_V)):.3f} V)")
    print(f"{label2} voltage source: {voltage2_source} (mean {float(np.mean(voltage2_V)):.3f} V)")
    print(f"Neutralization model: {neutralization}")
    if str(neutralization).lower() in ("sodium_charge_exchange", "sodium_collision", "charge_exchange"):
        print(f"  sodium collision branch: {sodium_collision_branch}")

    return {
        "nu0_GHz": nu0,
        "center1_GHz": float(center1),
        "center1_fit_unc_GHz": float(dcenter1_fit),
        "center1_voltage_unc_GHz": float(dcenter1_V),
        "center1_total_unc_GHz": float(dcenter1_total),
        "center2_GHz": float(center2),
        "center2_fit_unc_GHz": float(dcenter2_fit),
        "center2_voltage_unc_GHz": float(dcenter2_V),
        "center2_total_unc_GHz": float(dcenter2_total),
        "isotope_shift_GHz": float(isotope_shift),
        "isotope_shift_fit_unc_GHz": float(isotope_shift_fit_unc),
        "isotope_shift_voltage_unc_GHz": float(isotope_shift_V_unc),
        "isotope_shift_total_unc_GHz": float(isotope_shift_total_unc),
        "tof_gate_us": tof_gate_us,
        "tof_col": tof_col,
        "show_tof_gate_plots": bool(show_tof_gate_plots),
        "frequency_multiplier": float(frequency_multiplier),
        "voltage_col": voltage_col,
        "voltage_multiplier": float(voltage_multiplier),
        "use_voltage_column": bool(use_voltage_column),
        "voltage1_source": voltage1_source,
        "voltage2_source": voltage2_source,
        "voltage1_mean_V": float(np.mean(voltage1_V)),
        "voltage2_mean_V": float(np.mean(voltage2_V)),
        "neutralization": neutralization,
        "sodium_mass_u": float(sodium_mass_u),
        "sodium_collision_branch": sodium_collision_branch,
        "num_points_1": int(cut_file_1.size),
        "num_points_2": int(cut_file_2.size),
    }


if __name__ == "__main__":
    print("Import this module in your notebook and call plot_two_isotopes_fit(...).")
