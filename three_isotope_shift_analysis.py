import numpy as np
import matplotlib.pyplot as plt
import satlas2


C = 299792458.0
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def gaussian(x, A, x0, sigma, y0):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + y0


class GaussianModel(satlas2.Model):
    def __init__(self, amplitude, center, sigma, background, name="Gaussian", prefunc=None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            "amplitude": satlas2.Parameter(value=amplitude, min=0.0, max=np.inf, vary=True),
            "center": satlas2.Parameter(value=center, vary=True),
            "sigma": satlas2.Parameter(value=sigma, min=1e-12, max=np.inf, vary=True),
            "background": satlas2.Parameter(value=background, vary=True),
        }

    def f(self, x):
        x = self.transform(x)
        amplitude = self.params["amplitude"].value
        center = self.params["center"].value
        sigma = self.params["sigma"].value
        background = self.params["background"].value
        return gaussian(x, amplitude, center, sigma, background)


def doppler_correct_ghz(nu_lab_ghz, mass_u, beam_voltage_V=10000.0, charge_e=1, geometry="collinear"):
    m = mass_u * AMU
    q = charge_e * E_CHARGE
    KE = q * beam_voltage_V

    beta = np.sqrt(2.0 * KE / (m * C**2))
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    if geometry == "collinear":
        factor = gamma * (1.0 - beta)
    elif geometry == "anticollinear":
        factor = gamma * (1.0 + beta)
    else:
        raise ValueError("geometry must be 'collinear' or 'anticollinear'")

    return np.asarray(nu_lab_ghz, dtype=float) * factor


def fit_histogram_peak(x, counts):
    i_max = np.argmax(counts)
    x0_guess = x[i_max]
    A_guess = counts[i_max] - np.min(counts)
    sigma_guess = max((x.max() - x.min()) / 20, 1e-6)
    y0_guess = np.min(counts)
    yerr = np.sqrt(np.asarray(counts, dtype=float))
    yerr[yerr <= 0] = 1.0

    model = GaussianModel(A_guess, x0_guess, sigma_guess, y0_guess)
    source = satlas2.Source(x, counts, yerr=yerr, name="HistogramData")
    source.addModel(model)

    fitter = satlas2.Fitter()
    fitter.addSource(source)
    fitter.fit(method="slsqp")

    params = model.params
    popt = np.array(
        [
            params["amplitude"].value,
            params["center"].value,
            params["sigma"].value,
            params["background"].value,
        ],
        dtype=float,
    )
    perr = np.array(
        [
            getattr(params["amplitude"], "unc", np.nan),
            getattr(params["center"], "unc", np.nan),
            getattr(params["sigma"], "unc", np.nan),
            getattr(params["background"], "unc", np.nan),
        ],
        dtype=float,
    )
    return popt, None, perr


def clean_numeric_column(dat, col):
    x = np.array(dat[col], dtype=float)
    return x[np.isfinite(x)]


def _fit_center_from_voltage(dat, mass_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0_ref):
    nu_lab = clean_numeric_column(dat, wn_col) * C * 100.0 * 1e-9
    nu = doppler_correct_ghz(nu_lab, mass_u, beam_voltage_V, charge_e, geometry)
    x = nu - nu0_ref

    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p, cov, err = fit_histogram_peak(centers, counts)
    return {
        "center": float(p[1]),
        "center_fit_unc": float(err[1]),
        "x": x,
        "counts": counts,
        "centers": centers,
        "fit_params": p,
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
    beam_voltage_V=10000.0,
    beam_voltage_unc_V=0.0,
    charge_e=1,
    geometry="collinear",
):
    nu32_lab = clean_numeric_column(cut_file_32S, wn_col) * C * 100.0 * 1e-9
    nu34_lab = clean_numeric_column(cut_file_34S, wn_col) * C * 100.0 * 1e-9
    nu36_lab = clean_numeric_column(cut_file_36S, wn_col) * C * 100.0 * 1e-9

    nu32 = doppler_correct_ghz(nu32_lab, mass32_u, beam_voltage_V, charge_e, geometry)
    nu34 = doppler_correct_ghz(nu34_lab, mass34_u, beam_voltage_V, charge_e, geometry)
    nu36 = doppler_correct_ghz(nu36_lab, mass36_u, beam_voltage_V, charge_e, geometry)

    nu0 = np.median(nu32)

    res32 = _fit_center_from_voltage(cut_file_32S, mass32_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0)
    res34 = _fit_center_from_voltage(cut_file_34S, mass34_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0)
    res36 = _fit_center_from_voltage(cut_file_36S, mass36_u, beam_voltage_V, wn_col, bins, charge_e, geometry, nu0)

    for res in (res32, res34, res36):
        res["center_voltage_unc"] = 0.0

    if beam_voltage_unc_V > 0:
        for dat, mass_u, res in [
            (cut_file_32S, mass32_u, res32),
            (cut_file_34S, mass34_u, res34),
            (cut_file_36S, mass36_u, res36),
        ]:
            c_plus = _fit_center_from_voltage(
                dat, mass_u, beam_voltage_V + beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
            )["center"]
            c_minus = _fit_center_from_voltage(
                dat, mass_u, beam_voltage_V - beam_voltage_unc_V, wn_col, bins, charge_e, geometry, nu0
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

    for ax, res, label, color in plot_info:
        xfit = np.linspace(res["centers"].min(), res["centers"].max(), 2000)
        yfit = gaussian(xfit, *res["fit_params"])

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
    }


if __name__ == "__main__":
    print("Import this module in your notebook and call plot_three_isotopes_fit(...).")
