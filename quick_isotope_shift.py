"""
Quick day-end isotope-shift analysis from collected sulfur CSV filenames.

This module is the lightweight orchestration layer around the existing
isotope_shift_analysis.py and three_isotope_shift_analysis.py fitters. It
loads files, infers isotope labels from filenames, runs the appropriate fit,
saves plots, and appends compact publication-library rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import isotope_shift_analysis as two_fit
import three_isotope_shift_analysis as three_fit

plt.show = lambda *args, **kwargs: None
FIT_BACKEND = "satlas2" if two_fit.satlas2 is not None else "scipy_curve_fit"
GHZ_TO_MHZ = 1000.0


SULFUR_MASSES_U = {
    "32S": 31.972071,
    "34S": 33.967867,
    "36S": 35.967081,
}

DEFAULT_ANALYSIS_OPTIONS: dict[str, Any] = {
    "wn_col": "wavemeter_wn1",
    "use_hene_calibration": True,
    "hene_col": "wavemeter_wn4",
    "hene_reference_wn": None,
    "hene_reference_wavelength_nm": 632.992,
    "hene_reference_wavelength_medium": "vacuum",
    "hene_wavenumber_medium": "vacuum",
    "frequency_multiplier": 2.0,
    "bin_width_MHz": 20.0,
    "tof_gate_us": None,
    "tof_col": "tof",
    "beam_voltage_V": 10000.0,
    "beam_voltage_unc_V": 0.0,
    "voltage_col": "voltage",
    "voltage_multiplier": 5962.49,
    "use_voltage_column": True,
    "charge_e": 1,
    "geometry": "collinear",
    "neutralization": "none",
    "sodium_collision_branch": "forward",
    "show_tof_gate_plots": False,
    "auto_remove_bad_scans": True,
    "bad_scan_min_coverage_fraction": 0.60,
    "bad_scan_min_points_fraction": 0.35,
    "bad_scan_max_spectrum_peak_z": 6.0,
    "use_bracketed_32S_reference": True,
    "bracket_max_shift_disagreement_MHz": 50.0,
    "validate_isotope_wavenumber": True,
    "isotope_wavenumber_windows": {
        "32S": [12625.16, 12625.22],
        "34S": [12624.86, 12624.93],
        "36S": [12624.48, 12624.72]
    },
}

LIBRARY_COLUMNS = [
    "analysis_id",
    "analysis_timestamp",
    "collection_date",
    "collection_time",
    "run_label",
    "transition",
    "isotopes",
    "files",
    "comparison",
    "nu0_MHz",
    "isotope_shift_MHz",
    "isotope_shift_total_unc_MHz",
    "isotope_shift_fit_unc_MHz",
    "isotope_shift_voltage_unc_MHz",
    "center_reference_MHz",
    "center_reference_total_unc_MHz",
    "center_comparison_MHz",
    "center_comparison_total_unc_MHz",
    "num_points_reference",
    "num_points_comparison",
    "plot_files",
    "fit_backend",
    "bad_scan_filter",
    "scans_removed",
    "points_removed",
    "options_json",
    "notes",
]


def load_cut_file(path: str | Path) -> np.ndarray:
    """Load a collected CSV file as a structured numpy array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def concatenate_cut_files(paths: list[str | Path]) -> np.ndarray:
    """Load and concatenate multiple files for the same isotope."""
    arrays = [load_cut_file(path) for path in paths]
    if not arrays:
        raise ValueError("At least one file is required.")
    if len(arrays) == 1:
        return arrays[0]
    names = arrays[0].dtype.names
    for path, array in zip(paths[1:], arrays[1:]):
        if array.dtype.names != names:
            raise ValueError(
                f"File {path} has columns {array.dtype.names}, expected {names}."
            )
    return np.concatenate(arrays)


def validate_file_isotope_wavenumber(
    files: list[Path],
    labels: list[str],
    *,
    options: dict[str, Any],
) -> None:
    if not options.get("validate_isotope_wavenumber", True):
        return
    windows = options.get("isotope_wavenumber_windows") or {}
    wn_col = options.get("wn_col", "wavemeter_wn1")
    problems = []
    for path, label in zip(files, labels):
        window = windows.get(label)
        if not window:
            continue
        dat = load_cut_file(path)
        if getattr(dat, "dtype", None) is None or dat.dtype.names is None or wn_col not in dat.dtype.names:
            continue
        values = np.asarray(dat[wn_col], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        median = float(np.median(values))
        low, high = [float(v) for v in window]
        if not (low <= median <= high):
            problems.append(f"{path.name} labeled {label} has median {wn_col}={median:.6f}, outside {low:.5f}-{high:.5f}")
    if problems:
        raise ValueError(
            "One or more files look mislabeled for this transition:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad <= 0:
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale <= 0:
            return np.zeros_like(values, dtype=float)
        return np.abs(values - median) / scale
    return 0.6745 * np.abs(values - median) / mad


def _scan_ids_from_bin_index(scan_bin_index: np.ndarray) -> np.ndarray:
    scan_bin_index = np.asarray(scan_bin_index, dtype=float)
    scan_ids = np.zeros(scan_bin_index.size, dtype=int)
    if scan_bin_index.size == 0:
        return scan_ids
    valid = np.isfinite(scan_bin_index)
    previous = scan_bin_index[0]
    current_scan = 0
    for idx in range(1, scan_bin_index.size):
        value = scan_bin_index[idx]
        if valid[idx] and np.isfinite(previous) and value < previous:
            current_scan += 1
        scan_ids[idx] = current_scan
        if valid[idx]:
            previous = value
    return scan_ids


def remove_bad_scans(
    dat: np.ndarray,
    *,
    scan_bin_col: str = "scan_bin_index",
    wn_col: str = "wavemeter_wn1",
    spectrum_peak_col: str = "spectrum_peak",
    min_coverage_fraction: float = 0.60,
    min_points_fraction: float = 0.35,
    max_spectrum_peak_z: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Remove whole scan passes that look incomplete or wildly off-normal.

    Scan passes are inferred from resets in scan_bin_index. A scan is removed
    if it has poor bin coverage, too few events, invalid wavemeter data, a
    collapsed wavemeter range, or a large robust outlier in median spectrum
    peak.
    """
    if (
        getattr(dat, "dtype", None) is None
        or dat.dtype.names is None
        or scan_bin_col not in dat.dtype.names
        or wn_col not in dat.dtype.names
    ):
        return dat, {
            "enabled": False,
            "reason": "missing scan-bin or wavemeter column",
            "scans_total": 0,
            "scans_removed": 0,
            "points_removed": 0,
        }

    scan_bins = np.asarray(dat[scan_bin_col], dtype=float)
    scan_ids = _scan_ids_from_bin_index(scan_bins)
    unique_scans = np.unique(scan_ids)
    if unique_scans.size <= 1:
        return dat, {
            "enabled": True,
            "reason": "single scan pass",
            "scans_total": int(unique_scans.size),
            "scans_removed": 0,
            "points_removed": 0,
        }

    full_bin_count = max(len(np.unique(scan_bins[np.isfinite(scan_bins)])), 1)
    metrics = []
    for scan_id in unique_scans:
        mask = scan_ids == scan_id
        wn = np.asarray(dat[wn_col][mask], dtype=float)
        bins = scan_bins[mask]
        finite_wn = wn[np.isfinite(wn)]
        if spectrum_peak_col in dat.dtype.names:
            spectrum_peak = np.asarray(dat[spectrum_peak_col][mask], dtype=float)
            spectrum_peak_median = float(np.nanmedian(spectrum_peak))
        else:
            spectrum_peak_median = 0.0
        metrics.append(
            {
                "scan_id": int(scan_id),
                "points": int(np.count_nonzero(mask)),
                "coverage": len(np.unique(bins[np.isfinite(bins)])) / full_bin_count,
                "finite_fraction": float(finite_wn.size / max(np.count_nonzero(mask), 1)),
                "wn_range": float(np.nanmax(finite_wn) - np.nanmin(finite_wn)) if finite_wn.size else 0.0,
                "spectrum_peak_median": spectrum_peak_median,
            }
        )

    points = np.array([item["points"] for item in metrics], dtype=float)
    wn_ranges = np.array([item["wn_range"] for item in metrics], dtype=float)
    peak_medians = np.array([item["spectrum_peak_median"] for item in metrics], dtype=float)
    median_points = max(float(np.nanmedian(points)), 1.0)
    median_wn_range = max(float(np.nanmedian(wn_ranges)), 0.0)
    peak_z = _robust_z(peak_medians)

    keep_scan_ids = []
    removed_reasons = {}
    for idx, item in enumerate(metrics):
        reasons = []
        if item["coverage"] < float(min_coverage_fraction):
            reasons.append("low_coverage")
        if item["points"] < float(min_points_fraction) * median_points:
            reasons.append("low_points")
        if item["finite_fraction"] < 0.95:
            reasons.append("invalid_wavemeter")
        if median_wn_range > 0 and item["wn_range"] < 0.35 * median_wn_range:
            reasons.append("collapsed_wavemeter_range")
        if max_spectrum_peak_z > 0 and peak_z[idx] > float(max_spectrum_peak_z):
            reasons.append("spectrum_peak_outlier")
        if reasons:
            removed_reasons[item["scan_id"]] = reasons
        else:
            keep_scan_ids.append(item["scan_id"])

    if not keep_scan_ids:
        return dat, {
            "enabled": True,
            "reason": "all scans failed filter; original data retained",
            "scans_total": int(unique_scans.size),
            "scans_removed": 0,
            "points_removed": 0,
            "removed_reasons": removed_reasons,
        }

    keep_mask = np.isin(scan_ids, keep_scan_ids)
    return dat[keep_mask], {
        "enabled": True,
        "reason": "applied",
        "scans_total": int(unique_scans.size),
        "scans_removed": int(unique_scans.size - len(keep_scan_ids)),
        "points_removed": int(dat.size - np.count_nonzero(keep_mask)),
        "removed_reasons": removed_reasons,
    }


def infer_isotope_label(path: str | Path) -> str:
    """Infer labels such as 32S from a filename."""
    match = re.search(r"(?<!\d)(3[246])\s*S(?![A-Za-z0-9])", Path(path).name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer isotope label from filename: {path}")
    return f"{match.group(1)}S"


def normalize_isotope_label(label: str) -> str:
    text = str(label).strip().upper().replace("-", "")
    match = re.fullmatch(r"(3[246])S?", text)
    if not match:
        raise ValueError(f"Invalid isotope label: {label}. Use 32S, 34S, or 36S.")
    return f"{match.group(1)}S"


def parse_isotope_labels(value: str | None) -> list[str] | None:
    if value in (None, "", "none", "None"):
        return None
    parts = [part.strip() for part in re.split(r"[\n;,]+", value) if part.strip()]
    return [normalize_isotope_label(part) for part in parts]


def parse_tof_gate(value: str | None) -> tuple[float, float] | None:
    if value in (None, "", "none", "None"):
        return None
    text = str(value).strip().replace("–", "-").replace("—", "-")
    parts = [part.strip() for part in re.split(r"\s*(?:,|-)\s*", text) if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use --tof-gate-us min,max, for example 4.25,5.5")
    return float(parts[0]), float(parts[1])


def parse_per_isotope_tof_gates(value: str | None) -> dict[str, tuple[float, float]] | None:
    if value in (None, "", "none", "None"):
        return None
    gates: dict[str, tuple[float, float]] = {}
    for item in re.split(r"[;\n]+", value):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Per-isotope ToF gates must look like 32S=4.2,5.5;34S=4.4,5.8")
        label_text, gate_text = item.split("=", 1)
        gates[normalize_isotope_label(label_text)] = parse_tof_gate(gate_text)
    return gates or None


def load_options(config_path: str | Path | None, overrides: argparse.Namespace) -> dict[str, Any]:
    options = dict(DEFAULT_ANALYSIS_OPTIONS)
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            options.update(json.load(handle))

    for key in DEFAULT_ANALYSIS_OPTIONS:
        value = getattr(overrides, key, None)
        if value is not None:
            options[key] = value
    return options


def _compact_options(options: dict[str, Any]) -> dict[str, Any]:
    compact = dict(options)
    if compact.get("tof_gate_us") is not None:
        compact["tof_gate_us"] = [float(v) for v in compact["tof_gate_us"]]
    if compact.get("per_isotope_tof_gates"):
        compact["per_isotope_tof_gates"] = {
            label: [float(v) for v in gate]
            for label, gate in compact["per_isotope_tof_gates"].items()
            if gate is not None
        }
    return compact


def _save_open_figures(plot_dir: Path, analysis_id: str) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for index, fig_number in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_number)
        path = plot_dir / f"{analysis_id}_plot{index}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        saved.append(str(path))
    plt.close("all")
    return saved


def _analysis_id(run_label: str | None, collection_date: str | None) -> str:
    base = run_label or collection_date or "isotope_shift"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{cleaned or 'analysis'}"


def build_consecutive_isotope_blocks(files: list[Path], labels: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path, label in zip(files, labels):
        if blocks and blocks[-1]["label"] == label:
            blocks[-1]["files"].append(path)
        else:
            blocks.append({"label": label, "files": [path]})
    return blocks


def build_single_file_blocks(files: list[Path], labels: list[str]) -> list[dict[str, Any]]:
    return [{"label": label, "files": [path]} for path, label in zip(files, labels)]


def _file_time_key(path: Path) -> float:
    match = re.search(r"scan_(\d{8})_(\d{6})", path.name)
    if match:
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        return stamp.timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _fit_options_for_run(options: dict[str, Any], per_isotope_tof_gates: dict[str, tuple[float, float]]) -> dict[str, Any]:
    fit_options = dict(options)
    for key in (
        "auto_remove_bad_scans",
        "bad_scan_min_coverage_fraction",
        "bad_scan_min_points_fraction",
        "bad_scan_max_spectrum_peak_z",
        "per_isotope_tof_gates",
        "use_bracketed_32S_reference",
        "bracket_max_shift_disagreement_MHz",
        "validate_isotope_wavenumber",
        "isotope_wavenumber_windows",
    ):
        fit_options.pop(key, None)
    if per_isotope_tof_gates:
        fit_options["tof_gate_us"] = None
    fit_options["show_tof_gate_plots"] = bool(fit_options.get("show_tof_gate_plots", False))
    return fit_options


def _prepare_cut_file_for_label(
    label: str,
    paths: list[Path],
    *,
    options: dict[str, Any],
    per_isotope_tof_gates: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    dat = concatenate_cut_files(paths)
    if options.get("auto_remove_bad_scans", True):
        dat, summary = remove_bad_scans(
            dat,
            scan_bin_col="scan_bin_index",
            wn_col=options.get("wn_col", "wavemeter_wn1"),
            min_coverage_fraction=options.get("bad_scan_min_coverage_fraction", 0.60),
            min_points_fraction=options.get("bad_scan_min_points_fraction", 0.35),
            max_spectrum_peak_z=options.get("bad_scan_max_spectrum_peak_z", 6.0),
        )
    else:
        summary = {
            "enabled": False,
            "reason": "disabled",
            "scans_total": 0,
            "scans_removed": 0,
            "points_removed": 0,
        }
    if label in per_isotope_tof_gates:
        dat = two_fit.apply_tof_gate(
            dat,
            tof_gate_us=per_isotope_tof_gates[label],
            tof_col=options.get("tof_col", "tof"),
        )
    return dat, summary


def _rest_frequencies_for_label(dat: np.ndarray, label: str, options: dict[str, Any]) -> np.ndarray:
    nu_lab, voltage_V, _ = two_fit._lab_frequency_and_voltage(
        dat,
        options.get("wn_col", "wavemeter_wn1"),
        frequency_multiplier=options.get("frequency_multiplier", 2.0),
        beam_voltage_V=options.get("beam_voltage_V", 10000.0),
        voltage_col=options.get("voltage_col", "voltage"),
        voltage_multiplier=options.get("voltage_multiplier", two_fit.B_HVD2),
        use_voltage_column=options.get("use_voltage_column", True),
        voltage_offset_V=options.get("voltage_offset_V", 0.0),
        use_hene_calibration=options.get("use_hene_calibration", False),
        hene_col=options.get("hene_col", "wavemeter_wn4"),
        hene_reference_wn=options.get("hene_reference_wn"),
        hene_reference_wavelength_nm=options.get("hene_reference_wavelength_nm", two_fit.DEFAULT_HENE_WAVELENGTH_NM),
        hene_reference_wavelength_medium=options.get("hene_reference_wavelength_medium", "vacuum"),
        hene_wavenumber_medium=options.get("hene_wavenumber_medium", "vacuum"),
    )
    return two_fit.doppler_correct_ghz(
        nu_lab,
        SULFUR_MASSES_U[label],
        voltage_V,
        options.get("charge_e", 1),
        options.get("geometry", "collinear"),
        neutralization=options.get("neutralization", "none"),
        sodium_mass_u=options.get("sodium_mass_u", two_fit.SODIUM_MASS_U),
        sodium_collision_branch=options.get("sodium_collision_branch", "forward"),
    )


def _fit_absolute_center(dat: np.ndarray, label: str, options: dict[str, Any]) -> dict[str, Any]:
    nu_ref = float(np.median(_rest_frequencies_for_label(dat, label, options)))
    center, dcenter, x, counts, centers, fit_params, x_fit_window, quality = two_fit._fit_center_from_voltage(
        dat,
        SULFUR_MASSES_U[label],
        options.get("beam_voltage_V", 10000.0),
        options.get("wn_col", "wavemeter_wn1"),
        120,
        options.get("charge_e", 1),
        options.get("geometry", "collinear"),
        nu_ref,
        bin_width_MHz=options.get("bin_width_MHz"),
        frequency_multiplier=options.get("frequency_multiplier", 2.0),
        voltage_col=options.get("voltage_col", "voltage"),
        voltage_multiplier=options.get("voltage_multiplier", two_fit.B_HVD2),
        use_voltage_column=options.get("use_voltage_column", True),
        voltage_offset_V=options.get("voltage_offset_V", 0.0),
        use_hene_calibration=options.get("use_hene_calibration", False),
        hene_col=options.get("hene_col", "wavemeter_wn4"),
        hene_reference_wn=options.get("hene_reference_wn"),
        hene_reference_wavelength_nm=options.get("hene_reference_wavelength_nm", two_fit.DEFAULT_HENE_WAVELENGTH_NM),
        hene_reference_wavelength_medium=options.get("hene_reference_wavelength_medium", "vacuum"),
        hene_wavenumber_medium=options.get("hene_wavenumber_medium", "vacuum"),
        neutralization=options.get("neutralization", "none"),
        sodium_mass_u=options.get("sodium_mass_u", two_fit.SODIUM_MASS_U),
        sodium_collision_branch=options.get("sodium_collision_branch", "forward"),
    )
    return {
        "center_abs_GHz": float(nu_ref + center),
        "center_fit_unc_GHz": float(dcenter),
        "nu_ref_GHz": nu_ref,
        "x_GHz": x,
        "counts": counts,
        "centers_GHz": centers,
        "fit_params": fit_params,
        "x_fit_window_GHz": x_fit_window,
        "fit_quality": quality,
        "num_points": int(dat.size),
    }


def _plot_bracketed_fit(
    before: dict[str, Any],
    comparison: dict[str, Any],
    after: dict[str, Any],
    *,
    comparison_label: str,
    reference_interp_GHz: float,
    isotope_shift_GHz: float,
    shift_disagreement_MHz: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    plot_rows = [
        (axes[0], before, "32S before", "C0"),
        (axes[1], comparison, comparison_label, "C1" if comparison_label == "34S" else "C2"),
        (axes[2], after, "32S after", "C0"),
    ]
    x_limits = []
    for ax, result, label, color in plot_rows:
        centers_MHz = (result["centers_GHz"] + result["nu_ref_GHz"] - reference_interp_GHz) * GHZ_TO_MHZ
        xfit = np.linspace(result["centers_GHz"].min(), result["centers_GHz"].max(), 2000)
        xfit_MHz = (xfit + result["nu_ref_GHz"] - reference_interp_GHz) * GHZ_TO_MHZ
        ax.errorbar(
            centers_MHz,
            result["counts"],
            yerr=np.sqrt(np.clip(result["counts"], 1.0, None)),
            fmt="o",
            ms=4,
            capsize=2,
            color=color,
            ecolor="black",
            label=label,
        )
        ax.plot(xfit_MHz, two_fit.voigt(xfit, *result["fit_params"]), color=color, lw=2)
        center_MHz = (result["center_abs_GHz"] - reference_interp_GHz) * GHZ_TO_MHZ
        ax.axvline(center_MHz, color=color, linestyle="--", label=f"center = {center_MHz:.1f} MHz")
        ax.axvline(0.0, color="black", linestyle=":", label="interpolated 32S")
        ax.set_ylabel("Counts", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend()
        x_limits.append(two_fit._occupied_xlim(centers_MHz / GHZ_TO_MHZ, result["counts"], result["x_GHz"]))
    axes[-1].set_xlabel("Corrected frequency relative to interpolated 32S (MHz)", fontweight="bold")
    all_centers = []
    for result in (before, comparison, after):
        all_centers.extend(((result["centers_GHz"] + result["nu_ref_GHz"] - reference_interp_GHz) * GHZ_TO_MHZ).tolist())
    if all_centers:
        axes[-1].set_xlim(min(all_centers) - 80.0, max(all_centers) + 80.0)
    fig.suptitle(
        f"Bracketed {comparison_label}-32S: shift = {isotope_shift_GHz * GHZ_TO_MHZ:.1f} MHz, "
        f"before/after disagreement = {shift_disagreement_MHz:.1f} MHz",
        fontweight="bold",
    )
    plt.tight_layout()


def _comparison_label(label1: str, label2: str) -> str:
    low, high = sorted((label1, label2), key=lambda label: int(label[:-1]))
    return f"{high}-{low}"


def _two_isotope_row_generic(
    result: dict[str, Any],
    *,
    label1: str,
    label2: str,
    **metadata: Any,
) -> dict[str, Any]:
    shared = _shared_metadata(**metadata)
    sign = 1.0
    low, high = sorted((label1, label2), key=lambda label: int(label[:-1]))
    if (label1, label2) != (low, high):
        sign = -1.0
    if label1 == low:
        low_center = result["center1_GHz"]
        low_center_unc = result["center1_total_unc_GHz"]
        high_center = result["center2_GHz"]
        high_center_unc = result["center2_total_unc_GHz"]
        low_points = result["num_points_1"]
        high_points = result["num_points_2"]
    else:
        low_center = result["center2_GHz"]
        low_center_unc = result["center2_total_unc_GHz"]
        high_center = result["center1_GHz"]
        high_center_unc = result["center1_total_unc_GHz"]
        low_points = result["num_points_2"]
        high_points = result["num_points_1"]
    row = _base_row(
        **shared,
        comparison=f"{high}-{low}",
        nu0_MHz=result["nu0_GHz"] * GHZ_TO_MHZ,
        isotope_shift_MHz=result["isotope_shift_GHz"] * GHZ_TO_MHZ * sign,
        isotope_shift_total_unc_MHz=result["isotope_shift_total_unc_GHz"] * GHZ_TO_MHZ,
        isotope_shift_fit_unc_MHz=result["isotope_shift_fit_unc_GHz"] * GHZ_TO_MHZ,
        isotope_shift_voltage_unc_MHz=result["isotope_shift_voltage_unc_GHz"] * GHZ_TO_MHZ,
        center_reference_MHz=low_center * GHZ_TO_MHZ,
        center_reference_total_unc_MHz=low_center_unc * GHZ_TO_MHZ,
        center_comparison_MHz=high_center * GHZ_TO_MHZ,
        center_comparison_total_unc_MHz=high_center_unc * GHZ_TO_MHZ,
        num_points_reference=low_points,
        num_points_comparison=high_points,
    )
    return _add_fit_quality(row, result.get("fit_quality", {}))


def run_analysis(
    files: list[str | Path],
    *,
    data_dir: str | Path | None = None,
    isotope_labels: list[str] | None = None,
    collection_date: str | None = None,
    collection_time: str | None = None,
    run_label: str | None = None,
    transition: str | None = None,
    notes: str | None = None,
    options: dict[str, Any] | None = None,
    plot_dir: str | Path = "analysis_plots",
    library_csv: str | Path | None = "data_library/isotope_shift_library.csv",
    library_jsonl: str | Path | None = "data_library/isotope_shift_library.jsonl",
    adjacent_single_scan_pairs: bool = False,
) -> list[dict[str, Any]]:
    """Run a two- or three-isotope fit from filenames and append library rows."""
    options = dict(DEFAULT_ANALYSIS_OPTIONS if options is None else options)
    data_dir_path = Path(data_dir) if data_dir else Path(".")
    resolved_files = [Path(path) if Path(path).is_absolute() else data_dir_path / Path(path) for path in files]

    if isotope_labels is not None:
        if len(isotope_labels) != len(resolved_files):
            raise ValueError("The number of isotope labels must match the number of files.")
        labels_for_files = [normalize_isotope_label(label) for label in isotope_labels]
    else:
        labels_for_files = [infer_isotope_label(path) for path in resolved_files]

    validate_file_isotope_wavenumber(resolved_files, labels_for_files, options=options)

    blocks = (
        build_single_file_blocks(resolved_files, labels_for_files)
        if adjacent_single_scan_pairs
        else build_consecutive_isotope_blocks(resolved_files, labels_for_files)
    )
    if len(blocks) > len(set(labels_for_files)) or len(blocks) > 2:
        if adjacent_single_scan_pairs and options.get("use_bracketed_32S_reference", True):
            rows = run_bracketed_block_analyses(
                blocks,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=transition,
                notes=notes,
                options=options,
                plot_dir=plot_dir,
            )
        else:
            rows = run_adjacent_block_analyses(
                blocks,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=transition,
                notes=notes,
                options=options,
                plot_dir=plot_dir,
            )
        if library_csv:
            append_library_csv(Path(library_csv), rows)
        if library_jsonl:
            append_library_jsonl(Path(library_jsonl), rows)
        return rows

    isotope_to_paths: dict[str, list[Path]] = {}
    for label, path in zip(labels_for_files, resolved_files):
        isotope_to_paths.setdefault(label, []).append(path)

    labels = sorted(isotope_to_paths, key=lambda label: int(label[:-1]))
    if labels not in (["32S", "34S"], ["32S", "34S", "36S"]):
        raise ValueError("Expected either 32S+34S or 32S+34S+36S files.")

    per_isotope_tof_gates = options.get("per_isotope_tof_gates") or {}
    per_isotope_tof_gates = {
        normalize_isotope_label(label): gate
        for label, gate in per_isotope_tof_gates.items()
        if gate is not None
    }
    cut_files = {}
    filter_summaries = {}
    for label, paths in isotope_to_paths.items():
        cut_files[label], filter_summaries[label] = _prepare_cut_file_for_label(
            label,
            paths,
            options=options,
            per_isotope_tof_gates=per_isotope_tof_gates,
        )

    analysis_id = _analysis_id(run_label, collection_date)
    fit_options = _fit_options_for_run(options, per_isotope_tof_gates)

    if labels == ["32S", "34S"]:
        result = two_fit.plot_two_isotopes_fit(
            cut_file_1=cut_files["32S"],
            cut_file_2=cut_files["34S"],
            mass1_u=SULFUR_MASSES_U["32S"],
            mass2_u=SULFUR_MASSES_U["34S"],
            label1="32S",
            label2="34S",
            **fit_options,
        )
        plot_files = _save_open_figures(Path(plot_dir), analysis_id)
        rows = [
            _two_isotope_row(
                result,
                analysis_id=analysis_id,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=transition,
                notes=notes,
                files=resolved_files,
                isotope_labels=labels_for_files,
                isotope_file_groups=isotope_to_paths,
                plot_files=plot_files,
                options=options,
                filter_summaries=filter_summaries,
            )
        ]
    else:
        result = three_fit.plot_three_isotopes_fit(
            cut_file_32S=cut_files["32S"],
            cut_file_34S=cut_files["34S"],
            cut_file_36S=cut_files["36S"],
            mass32_u=SULFUR_MASSES_U["32S"],
            mass34_u=SULFUR_MASSES_U["34S"],
            mass36_u=SULFUR_MASSES_U["36S"],
            **fit_options,
        )
        plot_files = _save_open_figures(Path(plot_dir), analysis_id)
        rows = [
            _three_isotope_row(
                result,
                comparison_label="34S-32S",
                comparison_isotope="34S",
                analysis_id=analysis_id,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=transition,
                notes=notes,
                files=resolved_files,
                isotope_labels=labels_for_files,
                isotope_file_groups=isotope_to_paths,
                plot_files=plot_files,
                options=options,
                filter_summaries=filter_summaries,
            ),
            _three_isotope_row(
                result,
                comparison_label="36S-32S",
                comparison_isotope="36S",
                analysis_id=analysis_id,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=transition,
                notes=notes,
                files=resolved_files,
                isotope_labels=labels_for_files,
                isotope_file_groups=isotope_to_paths,
                plot_files=plot_files,
                options=options,
                filter_summaries=filter_summaries,
            ),
        ]

    if library_csv:
        append_library_csv(Path(library_csv), rows)
    if library_jsonl:
        append_library_jsonl(Path(library_jsonl), rows)
    return rows


def run_adjacent_block_analyses(
    blocks: list[dict[str, Any]],
    *,
    collection_date: str | None,
    collection_time: str | None,
    run_label: str | None,
    transition: str | None,
    notes: str | None,
    options: dict[str, Any],
    plot_dir: str | Path,
) -> list[dict[str, Any]]:
    per_isotope_tof_gates = options.get("per_isotope_tof_gates") or {}
    per_isotope_tof_gates = {
        normalize_isotope_label(label): gate
        for label, gate in per_isotope_tof_gates.items()
        if gate is not None
    }
    fit_options = _fit_options_for_run(options, per_isotope_tof_gates)
    rows: list[dict[str, Any]] = []
    pair_index = 0
    reference_block: dict[str, Any] | None = None
    previous_label: str | None = None
    for block in blocks:
        label = block["label"]
        if label == "32S":
            reference_block = block
            previous_label = label
            continue
        if reference_block is None:
            previous_label = label
            continue
        if label not in ("34S", "36S"):
            previous_label = label
            continue
        if label == previous_label:
            previous_label = label
            continue

        pair_index += 1
        label_left = "32S"
        label_right = label
        analysis_id = _analysis_id(f"{run_label or collection_date}_pair{pair_index}_{label_left}_{label_right}", collection_date)
        try:
            dat_left, summary_left = _prepare_cut_file_for_label(
                label_left,
                reference_block["files"],
                options=options,
                per_isotope_tof_gates=per_isotope_tof_gates,
            )
            dat_right, summary_right = _prepare_cut_file_for_label(
                label_right,
                block["files"],
                options=options,
                per_isotope_tof_gates=per_isotope_tof_gates,
            )
            result = two_fit.plot_two_isotopes_fit(
                cut_file_1=dat_left,
                cut_file_2=dat_right,
                mass1_u=SULFUR_MASSES_U[label_left],
                mass2_u=SULFUR_MASSES_U[label_right],
                label1=label_left,
                label2=label_right,
                **fit_options,
            )
        except Exception as exc:
            plt.close("all")
            print(f"Skipping {label_right}-{label_left} pair: {exc}")
            continue
        plot_files = _save_open_figures(Path(plot_dir), analysis_id)
        files = list(reference_block["files"]) + list(block["files"])
        rows.append(
            _two_isotope_row_generic(
                result,
                label1=label_left,
                label2=label_right,
                analysis_id=analysis_id,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=f"{run_label or ''} pair {pair_index}".strip(),
                transition=transition,
                notes=notes,
                files=files,
                isotope_labels=[label_left] * len(reference_block["files"]) + [label_right] * len(block["files"]),
                isotope_file_groups={label_left: list(reference_block["files"]), label_right: list(block["files"])},
                plot_files=plot_files,
                options=options,
                filter_summaries={label_left: summary_left, label_right: summary_right},
            )
        )
        previous_label = label
    if not rows:
        raise ValueError("No adjacent isotope block pairs were found.")
    return rows


def run_bracketed_block_analyses(
    blocks: list[dict[str, Any]],
    *,
    collection_date: str | None,
    collection_time: str | None,
    run_label: str | None,
    transition: str | None,
    notes: str | None,
    options: dict[str, Any],
    plot_dir: str | Path,
) -> list[dict[str, Any]]:
    blocks = sorted(blocks, key=lambda block: _file_time_key(block["files"][0]))
    per_isotope_tof_gates = options.get("per_isotope_tof_gates") or {}
    per_isotope_tof_gates = {
        normalize_isotope_label(label): gate
        for label, gate in per_isotope_tof_gates.items()
        if gate is not None
    }
    max_disagreement = float(options.get("bracket_max_shift_disagreement_MHz", 50.0))
    rows: list[dict[str, Any]] = []
    pair_index = 0

    for idx, block in enumerate(blocks):
        label = block["label"]
        if label not in ("34S", "36S"):
            continue
        before = next((item for item in reversed(blocks[:idx]) if item["label"] == "32S"), None)
        after = next((item for item in blocks[idx + 1:] if item["label"] == "32S"), None)
        if before is None or after is None:
            print(f"Skipping {label}: no bracketing 32S scan before and after.")
            continue

        pair_index += 1
        analysis_id = _analysis_id(f"{run_label or collection_date}_bracket{pair_index}_32S_{label}", collection_date)
        try:
            dat_before, summary_before = _prepare_cut_file_for_label(
                "32S",
                before["files"],
                options=options,
                per_isotope_tof_gates=per_isotope_tof_gates,
            )
            dat_comparison, summary_comparison = _prepare_cut_file_for_label(
                label,
                block["files"],
                options=options,
                per_isotope_tof_gates=per_isotope_tof_gates,
            )
            dat_after, summary_after = _prepare_cut_file_for_label(
                "32S",
                after["files"],
                options=options,
                per_isotope_tof_gates=per_isotope_tof_gates,
            )
            result_before = _fit_absolute_center(dat_before, "32S", options)
            result_comparison = _fit_absolute_center(dat_comparison, label, options)
            result_after = _fit_absolute_center(dat_after, "32S", options)
        except Exception as exc:
            plt.close("all")
            print(f"Skipping bracketed {label}-32S pair: {exc}")
            continue

        t_before = _file_time_key(before["files"][0])
        t_comparison = _file_time_key(block["files"][0])
        t_after = _file_time_key(after["files"][0])
        fraction = (t_comparison - t_before) / max(t_after - t_before, 1.0)
        fraction = min(max(fraction, 0.0), 1.0)
        reference_center = (
            (1.0 - fraction) * result_before["center_abs_GHz"]
            + fraction * result_after["center_abs_GHz"]
        )
        reference_unc = float(
            np.sqrt(
                ((1.0 - fraction) * result_before["center_fit_unc_GHz"]) ** 2
                + (fraction * result_after["center_fit_unc_GHz"]) ** 2
            )
        )
        isotope_shift = result_comparison["center_abs_GHz"] - reference_center
        shift_before = result_comparison["center_abs_GHz"] - result_before["center_abs_GHz"]
        shift_after = result_comparison["center_abs_GHz"] - result_after["center_abs_GHz"]
        shift_disagreement_MHz = abs(shift_before - shift_after) * GHZ_TO_MHZ
        fit_unc = float(np.sqrt(result_comparison["center_fit_unc_GHz"] ** 2 + reference_unc ** 2))
        nu0 = 0.5 * (reference_center + result_comparison["center_abs_GHz"])

        _plot_bracketed_fit(
            result_before,
            result_comparison,
            result_after,
            comparison_label=label,
            reference_interp_GHz=reference_center,
            isotope_shift_GHz=isotope_shift,
            shift_disagreement_MHz=shift_disagreement_MHz,
        )
        plot_files = _save_open_figures(Path(plot_dir), analysis_id)
        files = list(before["files"]) + list(block["files"]) + list(after["files"])
        filter_summaries = {
            "32S_before": summary_before,
            label: summary_comparison,
            "32S_after": summary_after,
        }
        shared = _shared_metadata(
            analysis_id=analysis_id,
            collection_date=collection_date,
            collection_time=collection_time,
            run_label=f"{run_label or ''} bracket {pair_index}".strip(),
            transition=transition,
            notes=notes,
            files=files,
            isotope_labels=["32S"] * len(before["files"]) + [label] * len(block["files"]) + ["32S"] * len(after["files"]),
            isotope_file_groups={"32S": list(before["files"]) + list(after["files"]), label: list(block["files"])},
            plot_files=plot_files,
            options=options,
            filter_summaries=filter_summaries,
        )
        row = _base_row(
            **shared,
            comparison=f"{label}-32S",
            nu0_MHz=nu0 * GHZ_TO_MHZ,
            isotope_shift_MHz=isotope_shift * GHZ_TO_MHZ,
            isotope_shift_total_unc_MHz=fit_unc * GHZ_TO_MHZ,
            isotope_shift_fit_unc_MHz=fit_unc * GHZ_TO_MHZ,
            isotope_shift_voltage_unc_MHz=0.0,
            center_reference_MHz=(reference_center - nu0) * GHZ_TO_MHZ,
            center_reference_total_unc_MHz=reference_unc * GHZ_TO_MHZ,
            center_comparison_MHz=(result_comparison["center_abs_GHz"] - nu0) * GHZ_TO_MHZ,
            center_comparison_total_unc_MHz=result_comparison["center_fit_unc_GHz"] * GHZ_TO_MHZ,
            num_points_reference=result_before["num_points"] + result_after["num_points"],
            num_points_comparison=result_comparison["num_points"],
        )
        row = _add_fit_quality(
            row,
            {
                "32S_before": result_before["fit_quality"],
                label: result_comparison["fit_quality"],
                "32S_after": result_after["fit_quality"],
                "bracket": {
                    "fraction": fraction,
                    "shift_before_MHz": shift_before * GHZ_TO_MHZ,
                    "shift_after_MHz": shift_after * GHZ_TO_MHZ,
                    "shift_disagreement_MHz": shift_disagreement_MHz,
                    "max_shift_disagreement_MHz": max_disagreement,
                    "passes": shift_disagreement_MHz <= max_disagreement,
                },
            },
        )
        rows.append(row)

    if not rows:
        raise ValueError("No bracketed isotope pairs were found. Add a 32S scan before and after each isotope scan.")
    return rows


def _base_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: "" for column in LIBRARY_COLUMNS}
    row.update(kwargs)
    row["analysis_timestamp"] = datetime.now().isoformat(timespec="seconds")
    return row


def _shared_metadata(
    *,
    analysis_id: str,
    collection_date: str | None,
    collection_time: str | None,
    run_label: str | None,
    transition: str | None,
    notes: str | None,
    files: list[Path],
    isotope_labels: list[str],
    isotope_file_groups: dict[str, list[Path]],
    plot_files: list[str],
    options: dict[str, Any],
    filter_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scans_removed = sum(int(item.get("scans_removed", 0)) for item in filter_summaries.values())
    points_removed = sum(int(item.get("points_removed", 0)) for item in filter_summaries.values())
    return {
        "analysis_id": analysis_id,
        "collection_date": collection_date or "",
        "collection_time": collection_time or "",
        "run_label": run_label or "",
        "transition": transition or "",
        "isotopes": ";".join(sorted(isotope_file_groups, key=lambda label: int(label[:-1]))),
        "files": ";".join(str(path) for path in files),
        "plot_files": ";".join(plot_files),
        "fit_backend": FIT_BACKEND,
        "bad_scan_filter": json.dumps(filter_summaries, sort_keys=True),
        "scans_removed": scans_removed,
        "points_removed": points_removed,
        "options_json": json.dumps(_compact_options(options), sort_keys=True),
        "notes": notes or "",
    }


def _add_fit_quality(row: dict[str, Any], fit_quality: dict[str, Any]) -> dict[str, Any]:
    try:
        quality_payload = json.loads(row.get("bad_scan_filter", "{}") or "{}")
    except json.JSONDecodeError:
        quality_payload = {}
    quality_payload["fit_quality"] = fit_quality
    row["bad_scan_filter"] = json.dumps(quality_payload, sort_keys=True)
    return row


def _two_isotope_row(result: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    shared = _shared_metadata(**metadata)
    row = _base_row(
        **shared,
        comparison="34S-32S",
        nu0_MHz=result["nu0_GHz"] * GHZ_TO_MHZ,
        isotope_shift_MHz=result["isotope_shift_GHz"] * GHZ_TO_MHZ,
        isotope_shift_total_unc_MHz=result["isotope_shift_total_unc_GHz"] * GHZ_TO_MHZ,
        isotope_shift_fit_unc_MHz=result["isotope_shift_fit_unc_GHz"] * GHZ_TO_MHZ,
        isotope_shift_voltage_unc_MHz=result["isotope_shift_voltage_unc_GHz"] * GHZ_TO_MHZ,
        center_reference_MHz=result["center1_GHz"] * GHZ_TO_MHZ,
        center_reference_total_unc_MHz=result["center1_total_unc_GHz"] * GHZ_TO_MHZ,
        center_comparison_MHz=result["center2_GHz"] * GHZ_TO_MHZ,
        center_comparison_total_unc_MHz=result["center2_total_unc_GHz"] * GHZ_TO_MHZ,
        num_points_reference=result["num_points_1"],
        num_points_comparison=result["num_points_2"],
    )
    return _add_fit_quality(row, result.get("fit_quality", {}))


def _three_isotope_row(
    result: dict[str, Any],
    *,
    comparison_label: str,
    comparison_isotope: str,
    **metadata: Any,
) -> dict[str, Any]:
    shared = _shared_metadata(**metadata)
    shift_key = f"shift_{comparison_isotope[:2]}_32_GHz"
    unc_key = f"shift_{comparison_isotope[:2]}_32_total_unc_GHz"
    reference = result["32S"]
    comparison = result[comparison_isotope]
    fit_unc = float(np.sqrt(reference["center_fit_unc"] ** 2 + comparison["center_fit_unc"] ** 2))
    voltage_unc = float(np.sqrt(reference["center_voltage_unc"] ** 2 + comparison["center_voltage_unc"] ** 2))
    row = _base_row(
        **shared,
        comparison=comparison_label,
        nu0_MHz=result["nu0_GHz"] * GHZ_TO_MHZ,
        isotope_shift_MHz=result[shift_key] * GHZ_TO_MHZ,
        isotope_shift_total_unc_MHz=result[unc_key] * GHZ_TO_MHZ,
        isotope_shift_fit_unc_MHz=fit_unc * GHZ_TO_MHZ,
        isotope_shift_voltage_unc_MHz=voltage_unc * GHZ_TO_MHZ,
        center_reference_MHz=reference["center"] * GHZ_TO_MHZ,
        center_reference_total_unc_MHz=reference["center_total_unc"] * GHZ_TO_MHZ,
        center_comparison_MHz=comparison["center"] * GHZ_TO_MHZ,
        center_comparison_total_unc_MHz=comparison["center_total_unc"] * GHZ_TO_MHZ,
        num_points_reference=result["num_points_32S"],
        num_points_comparison=result[f"num_points_{comparison_isotope}"],
    )
    return _add_fit_quality(
        row,
        {
            "32S": reference.get("fit_quality", {}),
            comparison_isotope: comparison.get("fit_quality", {}),
        },
    )


def append_library_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if path.exists():
        existing_rows = read_library_csv(path)
        existing_columns = list(existing_rows[0].keys()) if existing_rows else []
        if existing_columns and existing_columns != LIBRARY_COLUMNS:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LIBRARY_COLUMNS)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({column: row.get(column, "") for column in LIBRARY_COLUMNS})
            write_header = False
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIBRARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_library_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_library_csv(path: str | Path = "data_library/isotope_shift_library.csv") -> list[dict[str, Any]]:
    """Read the spectrum library CSV if it exists."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def library_row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("collection_date", "")),
        str(row.get("collection_time", "")),
        str(row.get("comparison", "")),
        str(row.get("files", "")),
    )


def append_new_library_rows_csv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append only rows that are not already represented in the CSV library."""
    existing_keys = {library_row_key(row) for row in read_library_csv(path)}
    new_rows = [row for row in rows if library_row_key(row) not in existing_keys]
    if new_rows:
        append_library_csv(path, new_rows)
    return new_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quick sulfur isotope-shift analysis from collected CSV filenames."
    )
    parser.add_argument("files", nargs="+", help="Collected CSV filenames. Isotopes are inferred from names like 32S_3-27-26.csv.")
    parser.add_argument("--data-dir", default=".", help="Directory containing the CSV files.")
    parser.add_argument("--isotopes", help="Optional isotope labels matching file order, e.g. 32S,34S,36S.")
    parser.add_argument("--collection-date", help="Date the data were collected, e.g. 2026-03-27.")
    parser.add_argument("--collection-time", help="Time/window of collection, e.g. 14:00-17:00 or afternoon.")
    parser.add_argument("--run-label", help="Short label for this analysis run.")
    parser.add_argument("--transition", help="Transition/line label for the library.")
    parser.add_argument("--notes", help="Free-form notes stored with the library row.")
    parser.add_argument("--config", help="JSON file with default analysis options.")
    parser.add_argument("--plot-dir", default="analysis_plots", help="Directory for saved fit plots.")
    parser.add_argument("--library-csv", default="data_library/isotope_shift_library.csv", help="CSV library path.")
    parser.add_argument("--library-jsonl", default="data_library/isotope_shift_library.jsonl", help="JSONL library path.")
    parser.add_argument("--no-library", action="store_true", help="Run analysis without appending library files.")

    parser.add_argument("--wn-col")
    parser.add_argument("--use-hene-calibration", dest="use_hene_calibration", action="store_true")
    parser.add_argument("--no-hene-calibration", dest="use_hene_calibration", action="store_false")
    parser.add_argument("--hene-col", dest="hene_col")
    parser.add_argument("--hene-reference-wn", dest="hene_reference_wn", type=float)
    parser.add_argument("--hene-reference-wavelength-nm", dest="hene_reference_wavelength_nm", type=float)
    parser.add_argument("--hene-reference-wavelength-medium", dest="hene_reference_wavelength_medium", choices=["vacuum", "air"])
    parser.add_argument("--hene-wavenumber-medium", dest="hene_wavenumber_medium", choices=["vacuum", "air"])
    parser.set_defaults(use_hene_calibration=None)
    parser.add_argument("--frequency-multiplier", dest="frequency_multiplier", type=float)
    parser.add_argument("--bin-width-MHz", dest="bin_width_MHz", type=float)
    parser.add_argument("--tof-gate-us", dest="tof_gate_us", type=parse_tof_gate)
    parser.add_argument("--tof-gate-32S", dest="tof_gate_32S", type=parse_tof_gate)
    parser.add_argument("--tof-gate-34S", dest="tof_gate_34S", type=parse_tof_gate)
    parser.add_argument("--tof-gate-36S", dest="tof_gate_36S", type=parse_tof_gate)
    parser.add_argument("--tof-col")
    parser.add_argument("--beam-voltage-V", dest="beam_voltage_V", type=float)
    parser.add_argument("--beam-voltage-unc-V", dest="beam_voltage_unc_V", type=float)
    parser.add_argument("--voltage-col")
    parser.add_argument("--voltage-multiplier", dest="voltage_multiplier", type=float)
    parser.add_argument("--voltage-offset-V", dest="voltage_offset_V", type=float)
    parser.add_argument("--fixed-voltage", dest="use_voltage_column", action="store_false")
    parser.add_argument("--charge-e", dest="charge_e", type=int)
    parser.add_argument("--geometry", choices=["collinear", "anticollinear"])
    parser.add_argument("--neutralization", choices=["none", "electron_capture", "sodium_charge_exchange"])
    parser.add_argument("--sodium-collision-branch", dest="sodium_collision_branch", choices=["forward", "momentum_transfer"])
    parser.add_argument("--show-tof-gate-plots", dest="show_tof_gate_plots", action="store_true")
    parser.add_argument("--no-bad-scan-filter", dest="auto_remove_bad_scans", action="store_false")
    parser.add_argument("--bad-scan-min-coverage-fraction", dest="bad_scan_min_coverage_fraction", type=float)
    parser.add_argument("--bad-scan-min-points-fraction", dest="bad_scan_min_points_fraction", type=float)
    parser.add_argument("--bad-scan-max-spectrum-peak-z", dest="bad_scan_max_spectrum_peak_z", type=float)
    parser.set_defaults(use_voltage_column=None, show_tof_gate_plots=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = load_options(args.config, args)
    if args.voltage_offset_V is not None:
        options["voltage_offset_V"] = args.voltage_offset_V
    per_isotope_tof_gates = {}
    for label, attr in [("32S", "tof_gate_32S"), ("34S", "tof_gate_34S"), ("36S", "tof_gate_36S")]:
        gate = getattr(args, attr, None)
        if gate is not None:
            per_isotope_tof_gates[label] = gate
    if per_isotope_tof_gates:
        options["per_isotope_tof_gates"] = per_isotope_tof_gates
    rows = run_analysis(
        args.files,
        data_dir=args.data_dir,
        isotope_labels=parse_isotope_labels(args.isotopes),
        collection_date=args.collection_date,
        collection_time=args.collection_time,
        run_label=args.run_label,
        transition=args.transition,
        notes=args.notes,
        options=options,
        plot_dir=args.plot_dir,
        library_csv=None if args.no_library else args.library_csv,
        library_jsonl=None if args.no_library else args.library_jsonl,
    )
    print()
    print("Saved library rows:")
    for row in rows:
        print(
            f"  {row['comparison']}: {float(row['isotope_shift_MHz']):.3f} +/- "
            f"{float(row['isotope_shift_total_unc_MHz']):.3f} MHz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
