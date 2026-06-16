"""Library-level uncertainty analysis for sulfur isotope-shift rows."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class InclusionCuts:
    comparison: str = "34S-32S"
    fit_unc_cut_MHz: float | None = 15.0
    total_unc_cut_MHz: float | None = None
    min_points_per_isotope: int = 100
    max_peak_to_model: float | None = 2.0
    require_bracket_pass: bool = True
    exclude_background: bool = True
    exclude_boundary: bool = True


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def weighted_mean_with_scatter_sem(values: np.ndarray, sigmas: np.ndarray) -> dict[str, float]:
    weights = 1.0 / np.square(sigmas)
    weighted_mean = float(np.sum(weights * values) / np.sum(weights))
    internal_unc = float(math.sqrt(1.0 / np.sum(weights)))
    v1 = float(np.sum(weights))
    v2 = float(np.sum(np.square(weights)))
    neff = float((v1 * v1) / v2) if v2 > 0 else float(values.size)
    denom = v1 - v2 / v1 if v1 > 0 else math.nan
    if values.size > 1 and denom > 0:
        weighted_var = float(np.sum(weights * np.square(values - weighted_mean)) / denom)
        weighted_std = float(math.sqrt(max(weighted_var, 0.0)))
        weighted_scatter_sem = float(weighted_std / math.sqrt(max(neff, 1.0)))
    else:
        weighted_std = 0.0
        weighted_scatter_sem = internal_unc
    chi2 = float(np.sum(weights * np.square(values - weighted_mean)))
    chi2_red = float(chi2 / max(values.size - 1, 1))
    unweighted_sem = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else internal_unc
    return {
        "N": int(values.size),
        "weighted_mean_MHz": weighted_mean,
        "internal_unc_MHz": internal_unc,
        "weighted_std_MHz": weighted_std,
        "weighted_scatter_sem_MHz": weighted_scatter_sem,
        "neff": neff,
        "chi2_red": chi2_red,
        "unweighted_mean_MHz": float(np.mean(values)),
        "unweighted_sem_MHz": unweighted_sem,
    }


_PAIR_SUFFIX_RE = re.compile(r"\s+(?:pair|bracket)\s+\d+\s*$", flags=re.IGNORECASE)


def _run_group_key(view: dict[str, Any]) -> tuple[str, str]:
    """Key identifying a single physical run, ignoring adjacent-pair/bracket splits."""
    label = str(view.get("run_label", "") or "").strip()
    prev = None
    while label and label != prev:
        prev = label
        label = _PAIR_SUFFIX_RE.sub("", label).strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    fallback = str(view.get("analysis_id", "")) or str(view.get("collection_time", ""))
    return (str(view.get("collection_date", "")), normalized or fallback)


def collapse_runs(included: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce rows that come from the same physical run to one independent point.

    Adjacent-pair/bracket rows share scans and systematics, so treating each as an
    independent measurement understates the combined uncertainty. Each group is
    collapsed to its inverse-variance weighted mean; the group sigma is floored at
    the typical single-pair fit uncertainty so it never shrinks below one
    measurement's precision, and is widened to the within-run scatter when the
    pairs disagree.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for view in included:
        key = _run_group_key(view)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(view)

    collapsed: list[dict[str, Any]] = []
    for key in order:
        members = groups[key]
        shifts = np.asarray([m["shift_MHz"] for m in members], dtype=float)
        sigmas = np.asarray([m["fit_unc_MHz"] for m in members], dtype=float)
        weights = 1.0 / np.square(sigmas)
        value = float(np.sum(weights * shifts) / np.sum(weights))
        typical_sigma = float(np.median(sigmas))
        within_rms = float(math.sqrt(np.mean(np.square(shifts - value)))) if shifts.size > 1 else 0.0
        group_sigma = max(typical_sigma, within_rms)
        collapsed.append(
            {
                "value_MHz": value,
                "sigma_MHz": group_sigma,
                "n_rows": int(shifts.size),
                "label": members[0].get("run_label") or members[0].get("collection_date") or key[1],
                "collection_date": members[0].get("collection_date", ""),
                "within_run_scatter_MHz": within_rms,
                "analysis_ids": [m.get("analysis_id", "") for m in members],
            }
        )
    return collapsed


def bayesian_random_effects_grid(values: np.ndarray, sigmas: np.ndarray) -> dict[str, Any]:
    """Grid approximation to the notebook's Normal(mu, sqrt(sigma_i^2 + tau^2)) model."""
    if values.size == 0:
        return {"available": False, "reason": "No included rows."}
    freq = weighted_mean_with_scatter_sem(values, sigmas)
    center = freq["weighted_mean_MHz"]
    spread = max(50.0, float(np.ptp(values)) + 6.0 * float(np.max(sigmas)))
    mu_grid = np.linspace(center - spread, center + spread, 801)
    tau_max = max(80.0, 3.0 * float(np.std(values, ddof=1)) if values.size > 1 else 80.0)
    tau_grid = np.linspace(0.0, tau_max, 401)

    mu = mu_grid[:, None]
    tau = tau_grid[None, :]
    var = np.square(sigmas)[None, None, :] + np.square(tau)[..., None]
    residual = values[None, None, :] - mu[..., None]
    loglike = -0.5 * np.sum(np.square(residual) / var + np.log(2.0 * np.pi * var), axis=2)
    log_mu_prior = -0.5 * np.square((mu_grid - center) / 50.0) - math.log(50.0 * math.sqrt(2.0 * math.pi))
    log_tau_prior = -0.5 * np.square(tau_grid / 25.0) + math.log(2.0 / (25.0 * math.sqrt(2.0 * math.pi)))
    logpost = loglike + log_mu_prior[:, None] + log_tau_prior[None, :]
    logpost -= float(np.max(logpost))
    posterior = np.exp(logpost)
    posterior_sum = float(np.sum(posterior))
    if not math.isfinite(posterior_sum) or posterior_sum <= 0:
        return {"available": False, "reason": "Bayesian grid posterior was numerically empty."}
    posterior /= posterior_sum

    mu_marginal = np.sum(posterior, axis=1)
    tau_marginal = np.sum(posterior, axis=0)
    mu_mean = float(np.sum(mu_grid * mu_marginal))
    mu_sd = float(math.sqrt(max(np.sum(np.square(mu_grid - mu_mean) * mu_marginal), 0.0)))
    tau_mean = float(np.sum(tau_grid * tau_marginal))
    cdf = np.cumsum(mu_marginal)
    low = float(np.interp(0.16, cdf, mu_grid))
    high = float(np.interp(0.84, cdf, mu_grid))
    density_area = float(np.trapezoid(mu_marginal, mu_grid)) if hasattr(np, "trapezoid") else float(np.trapz(mu_marginal, mu_grid))
    return {
        "available": True,
        "method": "grid_random_effects",
        "mu_mean_MHz": mu_mean,
        "mu_sd_MHz": mu_sd,
        "mu_16_MHz": low,
        "mu_84_MHz": high,
        "sigma_extra_mean_MHz": tau_mean,
        "mu_grid_MHz": mu_grid.tolist(),
        "mu_density": (mu_marginal / max(density_area, 1e-300)).tolist(),
    }


def _iter_peak_to_model_values(fit_quality: Any) -> list[float]:
    if isinstance(fit_quality, str):
        try:
            fit_quality = json.loads(fit_quality)
        except json.JSONDecodeError:
            return []
    if not isinstance(fit_quality, dict):
        return []
    values: list[float] = []
    stack = [fit_quality]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "peak_to_model":
                    found = safe_float(value)
                    if math.isfinite(found):
                        values.append(found)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return values


def _bracket_reason(row: dict[str, Any]) -> str | None:
    raw = row.get("options_json", "")
    try:
        options = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError:
        options = {}
    bracket = options.get("bracketed_32S_reference") or options.get("bracketed_reference")
    if not isinstance(bracket, dict):
        return None
    passes = bracket.get("passes")
    disagreement = safe_float(bracket.get("shift_disagreement_MHz"))
    limit = safe_float(bracket.get("max_shift_disagreement_MHz"))
    if passes is False:
        return "bracketed 32S reference failed"
    if math.isfinite(disagreement) and math.isfinite(limit) and disagreement > limit:
        return f"bracket disagreement {disagreement:.1f} > {limit:.1f} MHz"
    return None


def _row_view(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    files = [Path(item).name for item in str(row.get("files", "")).split(";") if item]
    return {
        "analysis_id": row.get("analysis_id", ""),
        "collection_date": row.get("collection_date", ""),
        "collection_time": row.get("collection_time", ""),
        "run_label": row.get("run_label", ""),
        "comparison": row.get("comparison", ""),
        "shift_MHz": finite_or_none(safe_float(row.get("isotope_shift_MHz"))),
        "total_unc_MHz": finite_or_none(safe_float(row.get("isotope_shift_total_unc_MHz"))),
        "fit_unc_MHz": finite_or_none(safe_float(row.get("isotope_shift_fit_unc_MHz"))),
        "voltage_unc_MHz": finite_or_none(safe_float(row.get("isotope_shift_voltage_unc_MHz"))),
        "num_points_reference": safe_int(row.get("num_points_reference")),
        "num_points_comparison": safe_int(row.get("num_points_comparison")),
        "files": files,
        "reasons": reasons,
    }


def analyze_library_uncertainty(rows: list[dict[str, Any]], cuts: InclusionCuts) -> dict[str, Any]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("comparison", "")) != cuts.comparison:
            continue
        reasons: list[str] = []
        run_text = f"{row.get('run_label', '')} {row.get('collection_time', '')}".lower()
        if cuts.exclude_background and ("background" in run_text or "_back" in run_text):
            reasons.append("background run")
        if cuts.exclude_boundary and "boundary" in run_text:
            reasons.append("boundary run")
        shift = safe_float(row.get("isotope_shift_MHz"))
        fit_unc = safe_float(row.get("isotope_shift_fit_unc_MHz"))
        total_unc = safe_float(row.get("isotope_shift_total_unc_MHz"))
        if not math.isfinite(shift):
            reasons.append("missing isotope shift")
        if not math.isfinite(fit_unc) or fit_unc <= 0:
            reasons.append("missing fit uncertainty")
        if not math.isfinite(total_unc) or total_unc <= 0:
            reasons.append("missing total uncertainty")
        if cuts.fit_unc_cut_MHz is not None and math.isfinite(fit_unc) and fit_unc > cuts.fit_unc_cut_MHz:
            reasons.append(f"fit uncertainty {fit_unc:.1f} > {cuts.fit_unc_cut_MHz:.1f} MHz")
        if cuts.total_unc_cut_MHz is not None and math.isfinite(total_unc) and total_unc > cuts.total_unc_cut_MHz:
            reasons.append(f"total uncertainty {total_unc:.1f} > {cuts.total_unc_cut_MHz:.1f} MHz")
        n_ref = safe_int(row.get("num_points_reference"))
        n_cmp = safe_int(row.get("num_points_comparison"))
        if n_ref < cuts.min_points_per_isotope or n_cmp < cuts.min_points_per_isotope:
            reasons.append(f"low gated points ({n_ref}, {n_cmp})")
        if cuts.max_peak_to_model is not None:
            peaks = _iter_peak_to_model_values(row.get("bad_scan_filter"))
            if peaks and max(peaks) > cuts.max_peak_to_model:
                reasons.append(f"peak/model {max(peaks):.2f} > {cuts.max_peak_to_model:.2f}")
        if cuts.require_bracket_pass:
            bracket = _bracket_reason(row)
            if bracket:
                reasons.append(bracket)
        view = _row_view(row, reasons)
        if reasons:
            excluded.append(view)
        else:
            included.append(view)

    groups = collapse_runs(included)
    values = np.asarray([group["value_MHz"] for group in groups], dtype=float)
    sigmas = np.asarray([group["sigma_MHz"] for group in groups], dtype=float)

    # Correlated HV systematic: the per-run voltage uncertainty stems from a common
    # beam-voltage uncertainty, so it does not average down with more runs. Carry the
    # representative (mean) per-run value as a floor added in quadrature to the
    # shared-shift uncertainty rather than folding it into the per-point weights.
    voltage_uncs = [
        float(row["voltage_unc_MHz"])
        for row in included
        if row.get("voltage_unc_MHz") is not None and math.isfinite(float(row["voltage_unc_MHz"]))
    ]
    systematic_unc = float(np.mean(voltage_uncs)) if voltage_uncs else 0.0

    result: dict[str, Any] = {
        "comparison": cuts.comparison,
        "cuts": {
            "fit_unc_cut_MHz": cuts.fit_unc_cut_MHz,
            "total_unc_cut_MHz": cuts.total_unc_cut_MHz,
            "min_points_per_isotope": cuts.min_points_per_isotope,
            "max_peak_to_model": cuts.max_peak_to_model,
            "require_bracket_pass": cuts.require_bracket_pass,
        },
        "included": included,
        "excluded": excluded,
        "groups": groups,
        "n_included_rows": len(included),
        "n_independent_runs": len(groups),
        "systematic_unc_MHz": systematic_unc,
        "frequentist": None,
        "bayesian": None,
    }
    if values.size:
        freq = weighted_mean_with_scatter_sem(values, sigmas)
        freq["systematic_unc_MHz"] = systematic_unc
        freq["total_unc_MHz"] = float(
            math.sqrt(freq["weighted_scatter_sem_MHz"] ** 2 + systematic_unc ** 2)
        )
        result["frequentist"] = freq
        bayes = bayesian_random_effects_grid(values, sigmas)
        if bayes.get("available"):
            bayes["systematic_unc_MHz"] = systematic_unc
            bayes["mu_total_unc_MHz"] = float(
                math.sqrt(bayes["mu_sd_MHz"] ** 2 + systematic_unc ** 2)
            )
        result["bayesian"] = bayes
    return result


def normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    return np.exp(-0.5 * np.square((x - mean) / sigma)) / (sigma * math.sqrt(2.0 * math.pi))


def plot_uncertainty_analysis(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    included = result.get("included", [])
    excluded = result.get("excluded", [])
    comparison = result.get("comparison", "")
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), gridspec_kw={"width_ratios": [1.5, 1.0, 0.85]})
    ax, dens, pull_ax = axes

    all_values = [
        row["shift_MHz"]
        for row in included + excluded
        if row.get("shift_MHz") is not None and math.isfinite(float(row["shift_MHz"]))
    ]
    if all_values:
        x_index = np.arange(len(included) + len(excluded))
        inc_x = np.arange(len(included))
        exc_x = np.arange(len(included), len(included) + len(excluded))
        if included:
            ax.errorbar(
                inc_x,
                [row["shift_MHz"] for row in included],
                yerr=[row["fit_unc_MHz"] for row in included],
                fmt="o",
                color="#1f9d45",
                ecolor="#333333",
                capsize=3,
                label="included",
            )
        if excluded:
            ax.errorbar(
                exc_x,
                [row["shift_MHz"] for row in excluded],
                yerr=[row["fit_unc_MHz"] or row["total_unc_MHz"] or 0.0 for row in excluded],
                fmt="o",
                color="#9a9a9a",
                ecolor="#b5b5b5",
                capsize=3,
                label="excluded",
            )
        labels = [row.get("collection_date") or row.get("analysis_id", "")[:8] for row in included + excluded]
        ax.set_xticks(x_index, labels, rotation=35, ha="right")
        ax.set_ylabel("Isotope shift (MHz)")
    else:
        ax.text(0.5, 0.5, "No matching rows", transform=ax.transAxes, ha="center", va="center")
        labels = []

    freq = result.get("frequentist") or {}
    bayes = result.get("bayesian") or {}
    groups = result.get("groups") or []
    sys_unc = float(result.get("systematic_unc_MHz", 0.0) or 0.0)
    if freq:
        mean = float(freq["weighted_mean_MHz"])
        sem = float(freq["weighted_scatter_sem_MHz"])
        freq_label = (
            f"freq {mean:.2f} +/- {sem:.2f} (stat) +/- {sys_unc:.2f} (sys)"
            if sys_unc > 0
            else f"freq {mean:.2f} +/- {sem:.2f}"
        )
        ax.axhline(mean, color="#313131", linestyle="--", linewidth=1.5, label=freq_label)
        ax.axhspan(mean - sem, mean + sem, color="#313131", alpha=0.12)
    if bayes and bayes.get("available"):
        mu = float(bayes["mu_mean_MHz"])
        sd = float(bayes["mu_sd_MHz"])
        bayes_label = (
            f"Bayes {mu:.2f} +/- {sd:.2f} (stat) +/- {sys_unc:.2f} (sys)"
            if sys_unc > 0
            else f"Bayes {mu:.2f} +/- {sd:.2f}"
        )
        ax.axhline(mu, color="#7b2cbf", linestyle="-.", linewidth=1.5, label=bayes_label)
        ax.axhspan(mu - sd, mu + sd, color="#7b2cbf", alpha=0.09)

    ax.set_title(f"{comparison} inclusion barrier")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Right panel: do the two methods agree on the shared shift? Both curves describe
    # the *mean* (not the run scatter): the frequentist sampling distribution of the
    # weighted mean vs the Bayesian posterior of mu. They should overlap.
    if freq:
        mean = float(freq["weighted_mean_MHz"])
        sem = float(freq["weighted_scatter_sem_MHz"])
        bayes_ok = bool(bayes and bayes.get("available"))
        mu_grid = np.asarray(bayes.get("mu_grid_MHz", []), dtype=float) if bayes_ok else np.asarray([])
        mu_density = np.asarray(bayes.get("mu_density", []), dtype=float) if bayes_ok else np.asarray([])

        spread_terms = [3.5 * sem]
        if bayes_ok:
            spread_terms.append(3.5 * float(bayes["mu_sd_MHz"]))
        pad = max(spread_terms) if spread_terms else 10.0
        x = np.linspace(mean - pad, mean + pad, 600)

        if bayes_ok and mu_grid.size and mu_density.size:
            dens.fill_between(mu_grid, mu_density, color="#1f9d45", alpha=0.30, label="Bayesian posterior")
            dens.plot(mu_grid, mu_density, color="#1f9d45", linewidth=1.2)
            dens.axvline(float(bayes["mu_mean_MHz"]), color="#1f9d45", linestyle=":", linewidth=1.2)
        dens.plot(
            x,
            normal_pdf(x, mean, sem),
            color="#c0392b",
            linewidth=1.8,
            label=f"frequentist {mean:.2f} +/- {sem:.2f}",
        )
        dens.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.0)
        dens.set_xlim(mean - pad, mean + pad)

        # Scatter-health annotation: chi2_red ~ 1 and small tau mean the run-to-run
        # scatter is consistent with the quoted per-run uncertainties.
        chi2_red = float(freq.get("chi2_red", float("nan")))
        note = f"chi2_red = {chi2_red:.2f}"
        if bayes_ok:
            note += f"\nextra scatter tau = {float(bayes['sigma_extra_mean_MHz']):.1f} MHz"
        dens.text(
            0.03,
            0.97,
            note,
            transform=dens.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85),
        )
    dens.set_title("Frequentist propagation vs Bayesian posterior")
    dens.set_xlabel(f"Shared isotope shift {comparison} (MHz)")
    dens.set_ylabel("Density")
    dens.legend(loc="best", fontsize=9)
    dens.grid(True, alpha=0.25)

    # Third panel: per-run pulls (residual / uncertainty). For trustworthy error bars
    # the pulls scatter within +/-1 with RMS ~ 1; outliers or an RMS well above 1 flag
    # underestimated uncertainties.
    if groups and freq:
        mean = float(freq["weighted_mean_MHz"])
        pulls = np.asarray(
            [(g["value_MHz"] - mean) / g["sigma_MHz"] for g in groups if g["sigma_MHz"] > 0],
            dtype=float,
        )
        idx = np.arange(pulls.size)
        pull_ax.axhspan(-1.0, 1.0, color="#1f9d45", alpha=0.12)
        pull_ax.axhline(0.0, color="#313131", linewidth=1.0)
        for level in (-2.0, 2.0):
            pull_ax.axhline(level, color="#999999", linestyle=":", linewidth=1.0)
        pull_ax.plot(idx, pulls, "o", color="#1f9d45", markersize=7)
        if pulls.size:
            rms = float(math.sqrt(np.mean(np.square(pulls))))
            pull_ax.text(
                0.5,
                0.98,
                f"pull RMS = {rms:.2f} (ideal ~1)",
                transform=pull_ax.transAxes,
                va="top",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85),
            )
        limit = max(2.6, float(np.max(np.abs(pulls))) * 1.15) if pulls.size else 2.6
        pull_ax.set_ylim(-limit, limit)
        pull_ax.set_xticks(idx)
        pull_ax.set_xlabel("independent run index")
    pull_ax.set_title("Per-run pulls")
    pull_ax.set_ylabel("normalized residual (pull)")
    pull_ax.grid(True, axis="y", alpha=0.25)

    n_rows = result.get("n_included_rows", len(included))
    n_runs = result.get("n_independent_runs", len(groups))
    subtitle = f"{n_rows} rows -> {n_runs} runs, {len(excluded)} excluded"
    fig.suptitle(f"{comparison} frequentist and Bayesian uncertainty analysis ({subtitle})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_plot_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "comparison"
