import numpy as np
import matplotlib.pyplot as plt
import satlas2
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz


C = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


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


def doppler_correct_ghz(nu_lab_ghz, mass_u, beam_voltage_V=10000.0, charge_e=1, geometry="collinear"):
    m = mass_u * AMU
    q = charge_e * E_CHARGE
    KE = q * beam_voltage_V

    gamma = 1.0 + KE / (m * C**2)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

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
):
    nu_lab = wn_to_lab_ghz(clean_numeric_column(dat, wn_col), frequency_multiplier=frequency_multiplier)
    nu = doppler_correct_ghz(nu_lab, mass_u, beam_voltage_V, charge_e, geometry)
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
    charge_e=1,
    geometry="collinear",
):
    if show_tof_gate_plots:
        plot_tof_gate_summary(cut_file_32S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="32S")
        plot_tof_gate_summary(cut_file_34S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="34S")
        plot_tof_gate_summary(cut_file_36S, tof_gate_us=tof_gate_us, tof_col=tof_col, bins=tof_plot_bins, label="36S")

    cut_file_32S = apply_tof_gate(cut_file_32S, tof_gate_us=tof_gate_us, tof_col=tof_col)
    cut_file_34S = apply_tof_gate(cut_file_34S, tof_gate_us=tof_gate_us, tof_col=tof_col)
    cut_file_36S = apply_tof_gate(cut_file_36S, tof_gate_us=tof_gate_us, tof_col=tof_col)

    nu32_lab = wn_to_lab_ghz(clean_numeric_column(cut_file_32S, wn_col), frequency_multiplier=frequency_multiplier)
    nu34_lab = wn_to_lab_ghz(clean_numeric_column(cut_file_34S, wn_col), frequency_multiplier=frequency_multiplier)
    nu36_lab = wn_to_lab_ghz(clean_numeric_column(cut_file_36S, wn_col), frequency_multiplier=frequency_multiplier)

    nu32 = doppler_correct_ghz(nu32_lab, mass32_u, beam_voltage_V, charge_e, geometry)
    nu34 = doppler_correct_ghz(nu34_lab, mass34_u, beam_voltage_V, charge_e, geometry)
    nu36 = doppler_correct_ghz(nu36_lab, mass36_u, beam_voltage_V, charge_e, geometry)

    nu0 = np.median(nu32)

    res32 = _fit_center_from_voltage(
        cut_file_32S, mass32_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier
    )
    res34 = _fit_center_from_voltage(
        cut_file_34S, mass34_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier
    )
    res36 = _fit_center_from_voltage(
        cut_file_36S, mass36_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0,
        bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier
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
                dat, mass_u, beam_voltage_V + beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0,
                bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier
            )["center"]
            c_minus = _fit_center_from_voltage(
                dat, mass_u, beam_voltage_V - beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0,
                bin_width_MHz=bin_width_MHz, frequency_multiplier=frequency_multiplier
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

        ax.step(res["centers"], res["counts"], where="mid", color=color, alpha=0.75, label=label)
        ax.plot(xfit, yfit, color=color, lw=2)
        ax.axvline(
            res["center"],
            color=color,
            linestyle="--",
            label=f"center = {res['center']:.3f} +/- {res['center_total_unc']:.3f} GHz",
        )
        ax.axvline(0.0, color="k", linestyle=":", label=r"$\nu_0$")
        ax.set_ylabel("Counts", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend()

    axes[-1].set_xlabel(r"Corrected frequency relative to $\nu_0$ (GHz)", fontweight="bold")
    axes[-1].set_xlim(x_left, x_right)
    fig.suptitle("Doppler-Corrected Sulfur Isotope Comparison", fontweight="bold", fontsize="x-large")
    plt.tight_layout()
    plt.show()

    print(f"32S center: {res32['center']:.6f} +/- {res32['center_total_unc']:.6f} GHz")
    print(f"34S center: {res34['center']:.6f} +/- {res34['center_total_unc']:.6f} GHz")
    print(f"36S center: {res36['center']:.6f} +/- {res36['center_total_unc']:.6f} GHz")
    print()
    print(f"Isotope shift (34S - 32S): {shift_34_32:.6f} +/- {shift_34_32_total_unc:.6f} GHz")
    print(f"  fit contribution: {shift_34_32_fit_unc:.6f} GHz")
    print(f"  voltage contribution: {shift_34_32_voltage_unc:.6f} GHz")
    print()
    print(f"Isotope shift (36S - 32S): {shift_36_32:.6f} +/- {shift_36_32_total_unc:.6f} GHz")
    print(f"  fit contribution: {shift_36_32_fit_unc:.6f} GHz")
    print(f"  voltage contribution: {shift_36_32_voltage_unc:.6f} GHz")
    print()
    print(f"nu0 reference: {nu0:.6f} GHz")

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
        "num_points_32S": int(cut_file_32S.size),
        "num_points_34S": int(cut_file_34S.size),
        "num_points_36S": int(cut_file_36S.size),
    }


if __name__ == "__main__":
    print("Import this module in your notebook and call plot_three_isotopes_fit(...).")
