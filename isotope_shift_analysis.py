import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


C = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def gaussian(x, amplitude, center, sigma, background):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background


def doppler_correct_ghz(nu_lab_ghz, mass_u, beam_voltage_V=10000.0, charge_e=1, geometry="collinear"):
    m = mass_u * AMU
    q = charge_e * E_CHARGE
    kinetic_energy = q * beam_voltage_V

    beta = np.sqrt(2.0 * kinetic_energy / (m * C**2))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    geometry = geometry.lower()
    if geometry == "collinear":
        factor = gamma * (1.0 - beta)
    elif geometry == "anticollinear":
        factor = gamma * (1.0 + beta)
    else:
        raise ValueError("geometry must be 'collinear' or 'anticollinear'")

    return np.asarray(nu_lab_ghz, dtype=float) * factor


def fit_histogram_peak(x, counts):
    i_max = np.argmax(counts)
    center_guess = x[i_max]
    amplitude_guess = counts[i_max] - np.min(counts)
    sigma_guess = max((x.max() - x.min()) / 20.0, 1e-6)
    background_guess = np.min(counts)

    p0 = [amplitude_guess, center_guess, sigma_guess, background_guess]
    popt, pcov = curve_fit(gaussian, x, counts, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def clean_numeric_column(dat, col):
    x = np.array(dat[col], dtype=float)
    return x[np.isfinite(x)]


def _fit_center_from_voltage(dat, mass_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0_ref):
    nu_lab = clean_numeric_column(dat, wn_col) * C * 100.0 * 1e-9
    nu_rest = doppler_correct_ghz(nu_lab, mass_u, beam_voltage_V, charge_e, geometry)
    x = nu_rest - nu0_ref

    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    popt, pcov, perr = fit_histogram_peak(centers, counts)
    return popt[1], perr[1], x, counts, centers, popt


def plot_two_isotopes_fit(
    cut_file_1,
    cut_file_2,
    mass1_u,
    mass2_u,
    wn_col="wavemeter_wn1",
    bins=120,
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=0.0,
    charge_e=1,
    geometry="collinear",
    label1="32S",
    label2="34S",
):
    """
    Compare two gated isotope spectra with Doppler correction and simple Gaussian fits.

    Returns fit centers and uncertainties for each isotope and the propagated isotope shift.
    """
    nu1_lab = clean_numeric_column(cut_file_1, wn_col) * C * 100.0 * 1e-9
    nu2_lab = clean_numeric_column(cut_file_2, wn_col) * C * 100.0 * 1e-9

    nu1 = doppler_correct_ghz(nu1_lab, mass1_u, beam_voltage_V, charge_e, geometry)
    nu2 = doppler_correct_ghz(nu2_lab, mass2_u, beam_voltage_V, charge_e, geometry)

    nu0 = 0.5 * (np.median(nu1) + np.median(nu2))

    center1, dcenter1_fit, x1, counts1, centers1, p1 = _fit_center_from_voltage(
        cut_file_1, mass1_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0
    )
    center2, dcenter2_fit, x2, counts2, centers2, p2 = _fit_center_from_voltage(
        cut_file_2, mass2_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0
    )

    isotope_shift = float(center2) - float(center1)
    isotope_shift_fit_unc = np.sqrt(dcenter1_fit**2 + dcenter2_fit**2)

    dcenter1_V = 0.0
    dcenter2_V = 0.0
    isotope_shift_V_unc = 0.0

    if beam_voltage_unc_V > 0:
        c1_plus, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_1, mass1_u, beam_voltage_V + beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
        )
        c1_minus, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_1, mass1_u, beam_voltage_V - beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
        )
        dcenter1_V = abs(float(c1_plus) - float(c1_minus)) / 2.0

        c2_plus, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_2, mass2_u, beam_voltage_V + beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
        )
        c2_minus, _, _, _, _, _ = _fit_center_from_voltage(
            cut_file_2, mass2_u, beam_voltage_V - beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
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
    yfit1 = gaussian(xfit1, *p1)
    yfit2 = gaussian(xfit2, *p2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].step(centers1, counts1, where="mid", color="C0", alpha=0.75, label=label1)
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
    axes[0].legend()

    axes[1].step(centers2, counts2, where="mid", color="C1", alpha=0.75, label=label2)
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
    axes[1].legend()

    fig.suptitle("Doppler-Corrected Isotope Comparison", fontweight="bold", fontsize="x-large")
    plt.tight_layout()
    plt.show()

    print(f"{label1} center: {center1:.6f} +/- {dcenter1_total:.6f} GHz")
    print(f"  fit contribution: {dcenter1_fit:.6f} GHz")
    print(f"  voltage contribution: {dcenter1_V:.6f} GHz")
    print(f"{label2} center: {center2:.6f} +/- {dcenter2_total:.6f} GHz")
    print(f"  fit contribution: {dcenter2_fit:.6f} GHz")
    print(f"  voltage contribution: {dcenter2_V:.6f} GHz")
    print(f"Isotope shift ({label2} - {label1}): {isotope_shift:.6f} +/- {isotope_shift_total_unc:.6f} GHz")
    print(f"  fit contribution: {isotope_shift_fit_unc:.6f} GHz")
    print(f"  voltage contribution: {isotope_shift_V_unc:.6f} GHz")
    print(f"nu0 reference: {nu0:.6f} GHz")

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
    }


if __name__ == "__main__":
    print("Import this module in your notebook and call plot_two_isotopes_fit(...).")
