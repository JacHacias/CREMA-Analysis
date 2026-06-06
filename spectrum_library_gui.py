"""
Local browser GUI for quick sulfur isotope-shift analysis and library review.

Run with:
    .\.venv\Scripts\python.exe .\spectrum_library_gui.py
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import mimetypes
import re
import shutil
import sys
import threading
import traceback
import webbrowser
import argparse
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

# Force a non-interactive backend before any project module imports pyplot, so
# plot generation is safe on the threaded HTTP server's worker threads.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quick_isotope_shift import (
    DEFAULT_ANALYSIS_OPTIONS,
    FIT_BACKEND,
    LIBRARY_COLUMNS,
    append_library_jsonl,
    append_new_library_rows_csv,
    infer_isotope_label,
    load_cut_file,
    parse_tof_gate,
    parse_isotope_labels,
    read_library_csv,
    run_analysis,
)
from centroid_stability_plot import plot_centroid_stability, plot_isotope_shift_stability
from library_uncertainty_analysis import (
    InclusionCuts,
    analyze_library_uncertainty,
    plot_uncertainty_analysis,
    safe_plot_label,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data")
LIBRARY_CSV = ROOT / "data_library" / "isotope_shift_library.csv"
LIBRARY_JSONL = ROOT / "data_library" / "isotope_shift_library.jsonl"
PLOT_DIR = ROOT / "analysis_plots"
SUMMARY_PLOT_DIR = ROOT / "analysis_plots" / "library_summary"
UNCERTAINTY_PLOT_DIR = ROOT / "uncertainty_plots"
EXPORTS_DIR = ROOT / "exports"
CONFIG_PATH = ROOT / "analysis_defaults.json"
MIN_LIBRARY_POINTS_PER_ISOTOPE = 100
MAX_LIBRARY_PEAK_TO_MODEL = 2.0
LIBRARY_SHIFT_WINDOWS_MHZ = {
    "34S-32S": (340.0, 455.0),
    "36S-32S": (600.0, 1000.0),
}

# matplotlib's pyplot state is process-global and not thread-safe; serialize all
# figure generation. A separate lock serializes read-modify-write of the library
# files so concurrent analyze/rebuild/delete requests cannot interleave.
PLOT_LOCK = threading.Lock()
MUTATION_LOCK = threading.Lock()


def load_default_options() -> dict:
    options = dict(DEFAULT_ANALYSIS_OPTIONS)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            options.update(json.load(handle))
    return options


def require_satlas_backend() -> None:
    if FIT_BACKEND != "satlas2":
        raise RuntimeError(
            "SATLAS2 is not available in the Python environment running this GUI. "
            "Restart with .venv\\Scripts\\python.exe spectrum_library_gui.py --no-browser "
            "so new library rows are not saved with scipy_curve_fit."
        )


def split_file_input(value: str) -> list[str]:
    parts = re.split(r"[\n;,]+", value)
    return [part.strip().strip('"') for part in parts if part.strip()]


def discover_folder_runs(data_dir: str | Path, include_background: bool = False, include_36s: bool = False) -> list[dict]:
    data_dir = Path(data_dir)
    isotope_re = re.compile(
        r"(?P<iso>3[246]S)_(?P<date>\d{1,2}-\d{1,2}-\d{2,4})(?P<suffix>.*)\.csv$",
        flags=re.IGNORECASE,
    )
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        match = isotope_re.match(path.name)
        if not match:
            continue
        suffix = match.group("suffix").lower()
        is_background = "back" in suffix or "background" in suffix
        if is_background and not include_background:
            continue
        key = (match.group("date"), "background" if is_background else "main")
        grouped.setdefault(key, {})[match.group("iso").upper()] = path

    runs = []
    for (date_label, kind), files_by_iso in sorted(grouped.items()):
        if "32S" not in files_by_iso or "34S" not in files_by_iso:
            continue
        files = [files_by_iso["32S"], files_by_iso["34S"]]
        if include_36s and "36S" in files_by_iso:
            files.append(files_by_iso["36S"])
        month, day, year = [int(part) for part in date_label.split("-")]
        if year < 100:
            year += 2000
        collection_date = f"{year:04d}-{month:02d}-{day:02d}"
        suffix = "_background" if kind == "background" else ""
        runs.append(
            {
                "files": files,
                "collection_date": collection_date,
                "collection_time": kind,
                "run_label": f"sulfur_{collection_date}{suffix}",
            }
        )
    return runs


def library_summary(limit: int = 100) -> dict:
    all_rows = read_library_csv(LIBRARY_CSV)
    rows = all_rows[-limit:]
    return {
        "path": str(LIBRARY_CSV),
        "count": len(all_rows),
        "rows": rows[::-1],
        "fit_backend": FIT_BACKEND,
    }


def plot_url(path: str) -> str:
    plot_path = Path(path)
    version = ""
    try:
        version = f"&v={int(plot_path.stat().st_mtime)}"
    except OSError:
        pass
    return f"/plot?path={quote(str(plot_path))}{version}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_optional_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def inclusion_cuts_from_query(query: dict[str, list[str]]) -> tuple[str, InclusionCuts]:
    comparison = query.get("comparison", ["34S-32S"])[0] or "34S-32S"
    cuts = InclusionCuts(
        comparison=comparison,
        fit_unc_cut_MHz=parse_optional_float(query.get("fit_unc_cut_MHz", ["15"])[0]),
        total_unc_cut_MHz=parse_optional_float(query.get("total_unc_cut_MHz", [""])[0]),
        min_points_per_isotope=int(float(query.get("min_points_per_isotope", ["100"])[0] or 100)),
        max_peak_to_model=parse_optional_float(query.get("max_peak_to_model", ["2.0"])[0]),
        require_bracket_pass=(query.get("require_bracket_pass", ["1"])[0] not in ("0", "false", "False", "")),
    )
    return comparison, cuts


def uncertainty_summary_from_query(query: dict[str, list[str]]) -> dict:
    comparison, cuts = inclusion_cuts_from_query(query)
    rows = read_library_csv(LIBRARY_CSV)
    result = analyze_library_uncertainty(rows, cuts)
    # Reuse a stable per-comparison filename so this directory does not grow
    # without bound; plot_url() appends an mtime version for cache-busting.
    UNCERTAINTY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    label = safe_plot_label(comparison)
    for stale in UNCERTAINTY_PLOT_DIR.glob(f"{label}_uncertainty_*.png"):
        try:
            stale.unlink()
        except OSError:
            pass
    plot_path = UNCERTAINTY_PLOT_DIR / f"{label}_uncertainty.png"
    try:
        with PLOT_LOCK:
            plot_uncertainty_analysis(result, plot_path)
        result["plot_url"] = plot_url(str(plot_path))
        result["plot_error"] = ""
    except OSError as exc:
        result["plot_url"] = ""
        result["plot_error"] = str(exc)
    if isinstance(result.get("bayesian"), dict):
        result["bayesian"].pop("mu_grid_MHz", None)
        result["bayesian"].pop("mu_density", None)
    return result


EXPORT_COLUMNS = [
    "collection_date",
    "collection_time",
    "run_label",
    "comparison",
    "shift_MHz",
    "fit_unc_MHz",
    "voltage_unc_MHz",
    "total_unc_MHz",
    "num_points_reference",
    "num_points_comparison",
]


def build_export_csv(comparison: str, result: dict) -> str:
    """Build a CSV of the included rows with the summary statistics as a header."""
    included = result.get("included", [])
    freq = result.get("frequentist", {}) or {}
    bayes = result.get("bayesian", {}) or {}
    buf = io.StringIO()
    buf.write("# CREMA isotope-shift export\n")
    buf.write(f"# comparison,{comparison}\n")
    buf.write(f"# included_rows,{len(included)}\n")
    buf.write(f"# excluded_rows,{len(result.get('excluded', []))}\n")
    buf.write(f"# weighted_mean_MHz,{freq.get('weighted_mean_MHz', '')}\n")
    buf.write(f"# weighted_scatter_sem_MHz,{freq.get('weighted_scatter_sem_MHz', '')}\n")
    buf.write(f"# chi2_red,{freq.get('chi2_red', '')}\n")
    if bayes.get("available"):
        buf.write(f"# bayesian_mu_MHz,{bayes.get('mu_mean_MHz', '')}\n")
        buf.write(f"# bayesian_mu_sd_MHz,{bayes.get('mu_sd_MHz', '')}\n")
        buf.write(f"# bayesian_sigma_extra_MHz,{bayes.get('sigma_extra_mean_MHz', '')}\n")
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in included:
        out = {column: row.get(column, "") for column in EXPORT_COLUMNS}
        if not out["comparison"]:
            out["comparison"] = comparison
        writer.writerow(out)
    return buf.getvalue()


def row_to_view(row: dict) -> dict:
    view = dict(row)
    plots = [item for item in str(row.get("plot_files", "")).split(";") if item]
    view["plot_urls"] = [plot_url(path) for path in plots]
    return view


def _write_library_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIBRARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LIBRARY_COLUMNS})


def _write_library_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _normalize_group_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _same_update_group(row: dict, collection_date: str, collection_time: str, run_label: str) -> bool:
    row_date = str(row.get("collection_date", "")).strip()
    row_time = str(row.get("collection_time", "")).strip()
    row_label = str(row.get("run_label", "")).strip()
    requested_label = str(run_label or "").strip()

    if requested_label and row_label:
        row_key = _normalize_group_text(row_label)
        requested_key = _normalize_group_text(requested_label)
        labels_match = row_key == requested_key or row_key.startswith(f"{requested_key}_pair_")
        dates_compatible = not collection_date or row_date == collection_date
        if labels_match and dates_compatible:
            return True

    if collection_date and row_date != collection_date:
        return False
    return row_time == str(collection_time or "").strip()


def _split_saved_list(value: str) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def _resolve_gui_files(files: list[str], data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [Path(item) if Path(item).is_absolute() else data_dir / item for item in files]


def _infer_label_from_wavenumber(path: Path, options: dict) -> str:
    windows = options.get("isotope_wavenumber_windows") or {}
    wn_col = options.get("wn_col", "wavemeter_wn1")
    dat = load_cut_file(path)
    if getattr(dat, "dtype", None) is None or dat.dtype.names is None or wn_col not in dat.dtype.names:
        raise ValueError(f"Could not infer isotope label from filename or wavemeter column: {path}")
    import numpy as np

    values = np.asarray(dat[wn_col], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Could not infer isotope label from empty wavemeter data: {path}")
    median = float(np.median(values))
    for label, window in windows.items():
        low, high = [float(item) for item in window]
        if low <= median <= high:
            return label
    raise ValueError(f"Could not match {path.name} median {wn_col}={median:.6f} to an isotope window.")


def infer_gui_file_labels(files: list[Path], options: dict) -> list[str]:
    labels = []
    for path in files:
        try:
            labels.append(infer_isotope_label(path))
        except ValueError:
            labels.append(_infer_label_from_wavenumber(path, options))
    return labels


def merge_with_existing_day_run(
    *,
    files: list[str],
    isotope_labels: list[str] | None,
    data_dir: str | Path,
    collection_date: str,
    collection_time: str,
    run_label: str,
    options: dict,
) -> tuple[list[Path], list[str], list[dict]]:
    existing_rows = read_library_csv(LIBRARY_CSV)
    matching_rows = [
        row for row in existing_rows
        if _same_update_group(row, collection_date, collection_time, run_label)
    ]
    new_files = _resolve_gui_files(files, data_dir)
    new_labels = isotope_labels or infer_gui_file_labels(new_files, options)
    if not matching_rows:
        return new_files, new_labels, []

    merged_files: list[Path] = []
    for row in matching_rows:
        merged_files.extend(Path(item) for item in _split_saved_list(row.get("files", "")))
    merged_files.extend(new_files)

    seen = set()
    unique_files = []
    for path in merged_files:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(path)

    merged_labels = infer_gui_file_labels(unique_files, options)
    return unique_files, merged_labels, matching_rows


def replace_library_group(old_rows: list[dict], new_rows: list[dict]) -> None:
    if not old_rows:
        new_rows_added = append_new_library_rows_csv(LIBRARY_CSV, new_rows)
        if new_rows_added:
            append_library_jsonl(LIBRARY_JSONL, new_rows_added)
        return

    old_ids = {row.get("analysis_id", "") for row in old_rows}
    old_plot_paths = [
        Path(path)
        for row in old_rows
        for path in _split_saved_list(row.get("plot_files", ""))
    ]
    kept_rows = [
        row for row in read_library_csv(LIBRARY_CSV)
        if row.get("analysis_id", "") not in old_ids
    ]
    updated_rows = kept_rows + new_rows
    _write_library_csv(LIBRARY_CSV, updated_rows)
    _write_library_jsonl(LIBRARY_JSONL, updated_rows)
    for path in old_plot_paths:
        try:
            resolved = path.resolve()
            resolved.relative_to(PLOT_DIR.resolve())
        except (ValueError, OSError):
            continue
        if resolved.exists():
            resolved.unlink()


def library_quality_reasons(
    row: dict,
    *,
    min_points: int = MIN_LIBRARY_POINTS_PER_ISOTOPE,
    max_peak_to_model: float = MAX_LIBRARY_PEAK_TO_MODEL,
) -> list[str]:
    """Return a list of reasons this row fails the library quality gates.

    An empty list means the row passes. Reasons are human-readable so they can
    be surfaced directly to the user instead of a bare pass/fail count.
    """
    reasons: list[str] = []
    try:
        n_reference = int(float(row.get("num_points_reference", 0)))
        n_comparison = int(float(row.get("num_points_comparison", 0)))
        shift = float(row.get("isotope_shift_MHz", 0))
    except (TypeError, ValueError):
        return ["Missing or non-numeric point counts / isotope shift."]

    comparison = str(row.get("comparison", ""))
    window = LIBRARY_SHIFT_WINDOWS_MHZ.get(comparison)
    if window and not (window[0] <= shift <= window[1]):
        reasons.append(
            f"Shift {shift:.1f} MHz outside {comparison} window "
            f"{window[0]:.0f}-{window[1]:.0f} MHz."
        )

    try:
        filter_summary = json.loads(row.get("bad_scan_filter", "{}") or "{}")
    except json.JSONDecodeError:
        filter_summary = {}
    fit_quality = filter_summary.get("fit_quality", {})
    bracket_quality = fit_quality.get("bracket")
    if isinstance(bracket_quality, dict):
        try:
            disagreement = float(bracket_quality.get("shift_disagreement_MHz", 0.0))
            maximum = float(bracket_quality.get("max_shift_disagreement_MHz", 50.0))
        except (TypeError, ValueError):
            reasons.append("Bracket disagreement values were non-numeric.")
        else:
            if disagreement > maximum:
                reasons.append(
                    f"Bracket shift disagreement {disagreement:.1f} MHz exceeds "
                    f"{maximum:.1f} MHz."
                )
    for name, quality in fit_quality.items():
        if not isinstance(quality, dict) or "peak_to_model_max" not in quality:
            continue
        try:
            peak_to_model = float(quality.get("peak_to_model_max", 1.0))
        except (TypeError, ValueError, AttributeError):
            continue
        if peak_to_model > max_peak_to_model:
            reasons.append(
                f"{name} peak/model {peak_to_model:.2f} exceeds {max_peak_to_model:.2f}."
            )

    if n_reference < min_points or n_comparison < min_points:
        reasons.append(
            f"Too few gated points ({n_reference}/{n_comparison} per isotope, "
            f"need {min_points})."
        )
    return reasons


def row_passes_library_quality(
    row: dict,
    *,
    min_points: int = MIN_LIBRARY_POINTS_PER_ISOTOPE,
    max_peak_to_model: float = MAX_LIBRARY_PEAK_TO_MODEL,
) -> bool:
    return not library_quality_reasons(
        row, min_points=min_points, max_peak_to_model=max_peak_to_model
    )


def delete_row_plots(rows: list[dict]) -> None:
    for row in rows:
        for path_text in _split_saved_list(row.get("plot_files", "")):
            path = Path(path_text)
            try:
                resolved = path.resolve()
                resolved.relative_to(PLOT_DIR.resolve())
            except (ValueError, OSError):
                continue
            if resolved.exists():
                resolved.unlink()


def format_float(value: str, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _timestamp_for_row(row: dict, index: int) -> datetime:
    date_text = str(row.get("collection_date", "")).strip()
    time_text = str(row.get("collection_time", "")).strip()
    if not date_text:
        date_text = str(row.get("analysis_timestamp", "")).split("T", 1)[0]

    try:
        base = datetime.fromisoformat(date_text)
    except ValueError:
        try:
            base = datetime.fromisoformat(str(row.get("analysis_timestamp", "")))
        except ValueError:
            base = datetime(2000, 1, 1)

    time_match = re.search(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})", time_text)
    if time_match:
        return base.replace(
            hour=int(time_match.group("hour")),
            minute=int(time_match.group("minute")),
            second=0,
            microsecond=0,
        )
    return base.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(minutes=index)


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _library_rows_for_centroid_plot(rows: list[dict]) -> list[dict]:
    entries = []
    for index, row in enumerate(rows):
        if row.get("comparison") != "34S-32S":
            continue
        center_32 = _float_or_none(row.get("center_reference_MHz"))
        center_34 = _float_or_none(row.get("center_comparison_MHz"))
        total_32 = _float_or_none(row.get("center_reference_total_unc_MHz"))
        total_34 = _float_or_none(row.get("center_comparison_total_unc_MHz"))
        if None in (center_32, center_34, total_32, total_34):
            continue
        entries.append(
            {
                "timestamp": _timestamp_for_row(row, index),
                "label": row.get("run_label") or row.get("collection_time") or row.get("collection_date"),
                "center_32_GHz": center_32 / 1000.0,
                "center_32_fit_unc_GHz": 0.0,
                "center_32_voltage_unc_GHz": 0.0,
                "center_32_total_unc_GHz": total_32 / 1000.0,
                "center_34_GHz": center_34 / 1000.0,
                "center_34_fit_unc_GHz": 0.0,
                "center_34_voltage_unc_GHz": 0.0,
                "center_34_total_unc_GHz": total_34 / 1000.0,
            }
        )
    return entries


def _library_rows_for_shift_plot(rows: list[dict], comparison: str) -> list[dict]:
    entries = []
    for index, row in enumerate(rows):
        if row.get("comparison") != comparison:
            continue
        shift = _float_or_none(row.get("isotope_shift_MHz"))
        fit_unc = _float_or_none(row.get("isotope_shift_fit_unc_MHz"))
        voltage_unc = _float_or_none(row.get("isotope_shift_voltage_unc_MHz"))
        total_unc = _float_or_none(row.get("isotope_shift_total_unc_MHz"))
        if None in (shift, fit_unc, voltage_unc, total_unc):
            continue
        entries.append(
            {
                "timestamp": _timestamp_for_row(row, index),
                "label": row.get("run_label") or row.get("collection_time") or row.get("collection_date"),
                "isotope_shift_GHz": shift / 1000.0,
                "isotope_shift_fit_unc_GHz": fit_unc / 1000.0,
                "isotope_shift_voltage_unc_GHz": voltage_unc / 1000.0,
                "isotope_shift_total_unc_GHz": total_unc / 1000.0,
            }
        )
    return entries


def _base_run_label(row: dict) -> str:
    label = str(row.get("run_label", "") or "").strip()
    while re.search(r"\s+pair\s+\d+\s*$", label, flags=re.IGNORECASE):
        label = re.sub(r"\s+pair\s+\d+\s*$", "", label, flags=re.IGNORECASE).strip()
    return label or str(row.get("collection_time", "") or row.get("collection_date", "")).strip()


def _shift_plot_entry_from_values(
    row: dict,
    index: int,
    *,
    shift_MHz: float,
    fit_unc_MHz: float,
    voltage_unc_MHz: float,
    total_unc_MHz: float,
) -> dict:
    return {
        "timestamp": _timestamp_for_row(row, index),
        "label": _base_run_label(row),
        "isotope_shift_GHz": shift_MHz / 1000.0,
        "isotope_shift_fit_unc_GHz": fit_unc_MHz / 1000.0,
        "isotope_shift_voltage_unc_GHz": voltage_unc_MHz / 1000.0,
        "isotope_shift_total_unc_GHz": total_unc_MHz / 1000.0,
    }


def _library_rows_for_32s_36s_shift_plot(rows: list[dict]) -> list[dict]:
    direct_entries = []
    grouped: dict[tuple[str, str], dict[str, tuple[int, dict]]] = {}

    for index, row in enumerate(rows):
        comparison = row.get("comparison", "")
        shift = _float_or_none(row.get("isotope_shift_MHz"))
        fit_unc = _float_or_none(row.get("isotope_shift_fit_unc_MHz"))
        voltage_unc = _float_or_none(row.get("isotope_shift_voltage_unc_MHz"))
        total_unc = _float_or_none(row.get("isotope_shift_total_unc_MHz"))
        if None in (shift, fit_unc, voltage_unc, total_unc):
            continue

        if comparison == "36S-32S":
            direct_entries.append(
                _shift_plot_entry_from_values(
                    row,
                    index,
                    shift_MHz=-shift,
                    fit_unc_MHz=fit_unc,
                    voltage_unc_MHz=voltage_unc,
                    total_unc_MHz=total_unc,
                )
            )
            continue

        if comparison in ("34S-32S", "36S-34S"):
            key = (str(row.get("collection_date", "")), _normalize_group_text(_base_run_label(row)))
            grouped.setdefault(key, {})[comparison] = (index, row)

    direct_keys = {
        (entry["timestamp"].date().isoformat(), _normalize_group_text(entry["label"]))
        for entry in direct_entries
    }
    derived_entries = []
    for key, pair_rows in grouped.items():
        if key in direct_keys or "34S-32S" not in pair_rows or "36S-34S" not in pair_rows:
            continue
        index_36_34, row_36_34 = pair_rows["36S-34S"]
        row_34_32 = pair_rows["34S-32S"][1]
        shift_34_32 = _float_or_none(row_34_32.get("isotope_shift_MHz"))
        shift_36_34 = _float_or_none(row_36_34.get("isotope_shift_MHz"))
        fit_34_32 = _float_or_none(row_34_32.get("isotope_shift_fit_unc_MHz"))
        fit_36_34 = _float_or_none(row_36_34.get("isotope_shift_fit_unc_MHz"))
        voltage_34_32 = _float_or_none(row_34_32.get("isotope_shift_voltage_unc_MHz"))
        voltage_36_34 = _float_or_none(row_36_34.get("isotope_shift_voltage_unc_MHz"))
        total_34_32 = _float_or_none(row_34_32.get("isotope_shift_total_unc_MHz"))
        total_36_34 = _float_or_none(row_36_34.get("isotope_shift_total_unc_MHz"))
        if None in (
            shift_34_32,
            shift_36_34,
            fit_34_32,
            fit_36_34,
            voltage_34_32,
            voltage_36_34,
            total_34_32,
            total_36_34,
        ):
            continue

        derived_entries.append(
            _shift_plot_entry_from_values(
                row_36_34,
                index_36_34,
                shift_MHz=-(shift_34_32 + shift_36_34),
                fit_unc_MHz=math.sqrt(fit_34_32**2 + fit_36_34**2),
                voltage_unc_MHz=math.sqrt(voltage_34_32**2 + voltage_36_34**2),
                total_unc_MHz=math.sqrt(total_34_32**2 + total_36_34**2),
            )
        )

    return sorted(direct_entries + derived_entries, key=lambda item: item["timestamp"])


def refresh_summary_plots() -> list[str]:
    with PLOT_LOCK:
        return _refresh_summary_plots_locked()


def _refresh_summary_plots_locked() -> list[str]:
    rows = read_library_csv(LIBRARY_CSV)
    SUMMARY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for old_plot in SUMMARY_PLOT_DIR.glob("*.png"):
        old_plot.unlink()
    saved: list[str] = []

    centroid_entries = _library_rows_for_centroid_plot(rows)
    if centroid_entries:
        fig, _ = plot_centroid_stability(
            centroid_entries,
            title="Total Centroid Stability",
            components=("total",),
            show_uncertainty_panel=False,
        )
        path = SUMMARY_PLOT_DIR / "total_centroid_stability.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    for comparison in sorted({row.get("comparison", "") for row in rows if row.get("comparison")}):
        shift_entries = _library_rows_for_shift_plot(rows, comparison)
        if not shift_entries:
            continue
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", comparison).strip("_")
        fig, _ = plot_isotope_shift_stability(
            shift_entries,
            title=f"{comparison} Isotope Shift Stability",
            components=("fit", "voltage", "total"),
            show_uncertainty_panel=True,
            show_average_sem=(comparison == "34S-32S"),
        )
        path = SUMMARY_PLOT_DIR / f"{label}_isotope_shift_stability.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    shift_32_36_entries = _library_rows_for_32s_36s_shift_plot(rows)
    if shift_32_36_entries:
        fig, _ = plot_isotope_shift_stability(
            shift_32_36_entries,
            title="32S-36S Isotope Shift Stability",
            components=("fit", "voltage", "total"),
            show_uncertainty_panel=True,
            show_average_sem=True,
        )
        path = SUMMARY_PLOT_DIR / "32S-36S_isotope_shift_stability.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def rebuild_library_from_folder(data_dir: str | Path, include_background: bool = False, include_36s: bool = False, options: dict | None = None) -> dict:
    options = load_default_options() if options is None else options
    runs = discover_folder_runs(data_dir, include_background=include_background, include_36s=include_36s)
    if not runs:
        raise ValueError(f"No 32S/34S file groups found in {data_dir}")

    LIBRARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if LIBRARY_CSV.exists():
        LIBRARY_CSV.unlink()
    if LIBRARY_JSONL.exists():
        LIBRARY_JSONL.unlink()
    for item in PLOT_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir() and item.name == "library_summary":
            shutil.rmtree(item, ignore_errors=True)

    rows_added = []
    for run in runs:
        rows = run_analysis(
            run["files"],
            data_dir=None,
            collection_date=run["collection_date"],
            collection_time=run["collection_time"],
            run_label=run["run_label"],
            transition="12625 cm-1 line",
            notes=f"Rebuilt automatically from {data_dir}.",
            options=options,
            plot_dir=PLOT_DIR,
            library_csv=None,
            library_jsonl=None,
        )
        new_rows = append_new_library_rows_csv(LIBRARY_CSV, rows)
        if new_rows:
            append_library_jsonl(LIBRARY_JSONL, new_rows)
        rows_added.extend(new_rows)

    refresh_summary_plots()
    return {"runs": runs, "rows": rows_added}


def options_from_payload(payload: dict) -> dict:
    options = load_default_options()
    per_isotope_tof_gates = dict(options.get("per_isotope_tof_gates") or {})
    for label, key in [("32S", "tof_gate_32S"), ("34S", "tof_gate_34S"), ("36S", "tof_gate_36S")]:
        if str(payload.get(key, "")).strip():
            per_isotope_tof_gates[label] = parse_tof_gate(payload[key])
    if per_isotope_tof_gates:
        options["per_isotope_tof_gates"] = per_isotope_tof_gates
        options["tof_gate_us"] = None
    elif str(payload.get("tof_gate_us", "")).strip():
        options["tof_gate_us"] = parse_tof_gate(payload["tof_gate_us"])
    if str(payload.get("bin_width_MHz", "")).strip():
        options["bin_width_MHz"] = float(payload["bin_width_MHz"])
    if str(payload.get("beam_voltage_unc_V", "")).strip():
        options["beam_voltage_unc_V"] = float(payload["beam_voltage_unc_V"])
    options["auto_remove_bad_scans"] = bool(payload.get("auto_remove_bad_scans"))
    return options


def rows_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No spectrum library rows yet.</p>'

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row.get('collection_date', ''))}</td>"
            f"<td>{html.escape(row.get('collection_time', ''))}</td>"
            f"<td>{html.escape(row.get('comparison', ''))}</td>"
            f"<td>{format_float(row.get('isotope_shift_MHz', ''), digits=3)}</td>"
            f"<td>{format_float(row.get('isotope_shift_total_unc_MHz', ''), digits=3)}</td>"
            f"<td>{html.escape(row.get('fit_backend', ''))}</td>"
            f"<td>{html.escape(row.get('scans_removed', '0') or '0')}</td>"
            f"<td>{html.escape(row.get('run_label', ''))}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Date</th><th>Time</th><th>Comparison</th>'
        '<th>Shift (MHz)</th><th>Unc. (MHz)</th><th>Fit</th><th>Bad Scans</th><th>Run</th>'
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_page() -> bytes:
    default_options = load_default_options()
    default_gates = default_options.get("per_isotope_tof_gates") or {}
    def gate_text(label: str) -> str:
        gate = default_gates.get(label)
        return f"{gate[0]},{gate[1]}" if gate else ""

    windows_text = ", ".join(
        f"{comparison} {low:.0f}-{high:.0f} MHz"
        for comparison, (low, high) in LIBRARY_SHIFT_WINDOWS_MHZ.items()
    )

    summary = library_summary(limit=50)
    rows = [row_to_view(row) for row in summary["rows"]]
    latest_plots = []
    for row in rows:
        for url in row.get("plot_urls", []):
            if url not in latest_plots:
                latest_plots.append(url)
        if len(latest_plots) >= 4:
            break
    plot_html = "".join(
        f'<figure><img src="{html.escape(url)}" alt="Saved spectrum fit plot"></figure>'
        for url in latest_plots
    ) or '<p class="empty">Run an analysis to preview saved fit plots here.</p>'
    # Stability summary plots are generated with matplotlib, which is the slow part
    # of a page load. Render an instant shell and let the JS hydrate them via
    # /api/library so the page is responsive immediately.
    summary_plot_html = '<p class="empty">Loading stability summary plots...</p>'

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CREMA Spectrum Library</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #182026;
      --muted: #60707d;
      --line: #d8e0e5;
      --panel: #f7f9fb;
      --accent: #0b7285;
      --accent-dark: #075864;
      --warn: #9a5a00;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font-size: 15px;
      line-height: 1.45;
    }}
    header {{
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #eef5f7;
    }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: var(--muted); }}
    main {{
      display: grid;
      grid-template-columns: minmax(340px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 88px);
    }}
    section {{ padding: 24px 28px; }}
    .control {{
      border-right: 1px solid var(--line);
      background: var(--panel);
    }}
    label {{
      display: block;
      margin: 14px 0 6px;
      font-weight: 650;
      font-size: 13px;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid #bac8d0;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
    }}
    textarea {{ min-height: 112px; resize: vertical; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    button.secondary {{ background: #425466; }}
    button:hover {{ background: var(--accent-dark); }}
    .actions {{ display: flex; gap: 10px; margin-top: 18px; align-items: center; }}
    .status {{
      margin-top: 16px;
      min-height: 48px;
      padding: 10px 12px;
      border-left: 4px solid var(--line);
      color: var(--muted);
      background: #fff;
    }}
    .status.ok {{ border-color: var(--accent); color: var(--ink); }}
    .status.err {{ border-color: #b42318; color: #7a271a; }}
    .library-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    h2 {{ margin: 0; font-size: 20px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{ background: #f1f5f8; font-size: 12px; color: #41515c; }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin: 18px 0 24px;
    }}
    .uncertainty-panel {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      margin: 0 0 24px;
    }}
    .compact-controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      align-items: end;
    }}
    .compact-controls label {{ margin-top: 0; }}
    .summary-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .summary-card {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 10px;
    }}
    .summary-card strong {{ display: block; font-size: 18px; }}
    .summary-card span {{ color: var(--muted); font-size: 12px; }}
    .tiny-table {{
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .tiny-table table {{ border: 0; }}
    .tiny-table th, .tiny-table td {{ font-size: 12px; }}
    figure {{ margin: 0; border: 1px solid var(--line); background: #fff; }}
    img {{ display: block; width: 100%; height: auto; }}
    .empty {{ color: var(--muted); margin: 12px 0; }}
    .help {{ color: var(--muted); font-size: 12px; margin-top: 5px; }}
    .row-actions {{ white-space: nowrap; }}
    .link-btn {{
      background: none;
      color: var(--accent);
      border: 1px solid var(--line);
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 6px;
    }}
    .link-btn:hover {{ background: var(--panel); color: var(--accent-dark); }}
    .link-btn.danger {{ color: #b42318; }}
    .link-btn.danger:hover {{ background: #fbeae8; color: #7a271a; }}
    button:disabled {{ opacity: 0.6; cursor: progress; }}
    .error-detail {{
      white-space: pre-wrap;
      font-size: 11px;
      max-height: 220px;
      overflow: auto;
      margin: 8px 0 0;
      color: #7a271a;
    }}
    details summary {{ cursor: pointer; font-size: 12px; margin-top: 6px; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      .control {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .grid2, .grid3 {{ grid-template-columns: 1fr; }}
      th, td {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CREMA Spectrum Library</h1>
    <p>Backend: {html.escape(FIT_BACKEND)} | Library: {html.escape(str(LIBRARY_CSV))}</p>
  </header>
  <main>
    <section class="control">
      <form id="analysis-form">
        <label for="data-dir">Data folder</label>
        <input id="data-dir" name="data_dir" value="{html.escape(str(DEFAULT_DATA_DIR))}">
        <div class="help">Relative filenames below are resolved from this folder.</div>

        <label for="files">Spectrum files</label>
        <textarea id="files" name="files" placeholder="32S_3-27-26.csv&#10;34S_3-27-26.csv&#10;36S_3-27-26.csv"></textarea>
        <div class="help">Paste full paths or filenames in collection order. Adjacent different-isotope scans are compared one pair at a time.</div>

        <label for="isotopes">Isotope labels, if filenames do not include them</label>
        <input id="isotopes" name="isotopes" placeholder="32S,34S or 32S,34S,36S">
        <div class="help">Use this for raw DAQ names like scan_20260506_140648.csv. Labels must match file order, including repeats.</div>

        <label><input type="checkbox" name="auto_remove_bad_scans" checked style="width:auto"> Auto-remove bad scan passes</label>
        <div class="help">Drops incomplete, low-count, invalid, or outlier scan passes before fitting.</div>

        <label><input type="checkbox" name="update_existing_day" style="width:auto"> Replace matching day/run entry</label>
        <div class="help">Optional. If checked, matching saved rows for the date/run label are replaced instead of appending new adjacent-pair rows.</div>

        <label><input type="checkbox" name="include_background" checked style="width:auto"> Include _back background files when rebuilding</label>
        <div class="help">The manual file box always uses exactly what you paste; this option only affects Rebuild From Folder.</div>

        <label><input type="checkbox" name="include_36s" style="width:auto"> Include 36S files when rebuilding</label>
        <div class="help">Leave unchecked to rebuild the old library without the March 27 36S scan.</div>

        <div class="grid2">
          <div>
            <label for="collection-date">Collection date</label>
            <input id="collection-date" name="collection_date" type="date">
          </div>
          <div>
            <label for="collection-time-start">Collection time (start / end)</label>
            <div class="grid2">
              <input id="collection-time-start" name="collection_time_start" type="time">
              <input id="collection-time-end" name="collection_time_end" type="time">
            </div>
          </div>
        </div>

        <label for="run-label">Run label</label>
        <input id="run-label" name="run_label" placeholder="sulfur_2026-03-27">

        <label for="transition">Transition</label>
        <input id="transition" name="transition" placeholder="12625 cm-1 line">

        <div class="grid3">
          <div>
            <label for="tof-gate-32">32S ToF gate us</label>
            <input id="tof-gate-32" name="tof_gate_32S" value="{html.escape(gate_text('32S'))}">
          </div>
          <div>
            <label for="tof-gate-34">34S ToF gate us</label>
            <input id="tof-gate-34" name="tof_gate_34S" value="{html.escape(gate_text('34S'))}">
          </div>
          <div>
            <label for="tof-gate-36">36S ToF gate us</label>
            <input id="tof-gate-36" name="tof_gate_36S" value="{html.escape(gate_text('36S'))}">
          </div>
        </div>

        <div class="grid2">
          <div>
            <label for="bin-width">Bin MHz</label>
            <input id="bin-width" name="bin_width_MHz" value="{default_options['bin_width_MHz']}">
          </div>
          <div>
            <label for="hv-unc">HV unc V</label>
            <input id="hv-unc" name="beam_voltage_unc_V" value="{default_options['beam_voltage_unc_V']}">
          </div>
        </div>

        <div class="grid2">
          <div>
            <label for="quality-min-points">Min gated points/isotope</label>
            <input id="quality-min-points" name="quality_min_points" value="{MIN_LIBRARY_POINTS_PER_ISOTOPE}">
          </div>
          <div>
            <label for="quality-max-peak">Max peak/model</label>
            <input id="quality-max-peak" name="quality_max_peak_to_model" value="{MAX_LIBRARY_PEAK_TO_MODEL}">
          </div>
        </div>
        <div class="help">Library quality gates applied before saving. Accepted shift windows: {html.escape(windows_text)}. Rejected rows are reported with reasons.</div>

        <label for="notes">Notes</label>
        <textarea id="notes" name="notes" style="min-height:74px"></textarea>

        <div class="actions">
          <button type="submit">Analyze and Save</button>
          <button class="secondary" type="button" id="rebuild">Rebuild From Folder</button>
          <button class="secondary" type="button" id="refresh">Refresh Library</button>
        </div>
        <div id="status" class="status">Ready.</div>
      </form>
    </section>
    <section>
      <div class="library-head">
        <h2>Library Stability Plots</h2>
        <div class="meta">Regenerated from saved rows</div>
      </div>
      <div id="summary-plots" class="plots">{summary_plot_html}</div>
      <div class="library-head">
        <h2>Uncertainty Analysis</h2>
        <div class="meta">Frequentist, Bayesian, and scan inclusion barrier</div>
      </div>
      <div class="uncertainty-panel">
        <div class="compact-controls">
          <div>
            <label for="unc-comparison">Comparison</label>
            <select id="unc-comparison">
              <option value="34S-32S">34S-32S</option>
              <option value="36S-32S">36S-32S</option>
              <option value="32S-36S">32S-36S</option>
            </select>
          </div>
          <div>
            <label for="unc-fit-cut">Max fit unc MHz</label>
            <input id="unc-fit-cut" value="15">
          </div>
          <div>
            <label for="unc-total-cut">Max total unc MHz</label>
            <input id="unc-total-cut" placeholder="optional">
          </div>
          <div>
            <label for="unc-min-points">Min points/isotope</label>
            <input id="unc-min-points" value="{MIN_LIBRARY_POINTS_PER_ISOTOPE}">
          </div>
          <div>
            <label for="unc-peak-model">Max peak/model</label>
            <input id="unc-peak-model" value="{MAX_LIBRARY_PEAK_TO_MODEL}">
          </div>
          <label><input type="checkbox" id="unc-bracket" checked style="width:auto"> Require bracket pass</label>
          <button type="button" id="unc-refresh">Run</button>
          <button type="button" class="secondary" id="unc-export">Export included set</button>
        </div>
        <div id="unc-summary" class="summary-cards"></div>
        <div id="unc-plot" class="plots"><p class="empty">Run the uncertainty analysis to draw the notebook-style summary.</p></div>
        <div id="unc-table" class="tiny-table"></div>
      </div>
      <div class="library-head">
        <h2>Recent Fit Plots</h2>
        <div class="meta">{summary['count']} saved library rows</div>
      </div>
      <div id="plots" class="plots">{plot_html}</div>
      <div class="library-head">
        <h2>Spectrum Library</h2>
        <div class="meta">Newest rows first</div>
      </div>
      <div id="library">{rows_table(summary['rows'])}</div>
    </section>
  </main>
  <script>
    const form = document.getElementById('analysis-form');
    const statusBox = document.getElementById('status');
    const library = document.getElementById('library');
    const plots = document.getElementById('plots');
    const summaryPlots = document.getElementById('summary-plots');
    const uncSummary = document.getElementById('unc-summary');
    const uncPlot = document.getElementById('unc-plot');
    const uncTable = document.getElementById('unc-table');

    function escapeHtml(value) {{
      return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    function setStatus(text, kind, detail) {{
      statusBox.className = 'status ' + (kind || '');
      if (detail) {{
        statusBox.textContent = '';
        const line = document.createElement('div');
        line.textContent = text;
        statusBox.appendChild(line);
        const det = document.createElement('details');
        const sum = document.createElement('summary');
        sum.textContent = 'Show error details';
        const pre = document.createElement('pre');
        pre.textContent = detail;
        pre.className = 'error-detail';
        det.appendChild(sum);
        det.appendChild(pre);
        statusBox.appendChild(det);
      }} else {{
        statusBox.textContent = text;
      }}
    }}

    async function withBusy(button, fn) {{
      const original = button ? button.textContent : '';
      if (button) {{ button.disabled = true; button.textContent = 'Working...'; }}
      try {{
        return await fn();
      }} finally {{
        if (button) {{ button.disabled = false; button.textContent = original; }}
      }}
    }}

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      const data = await response.json();
      if (!response.ok) {{ const err = new Error(data.error || 'Request failed'); err.detail = data.detail; throw err; }}
      return data;
    }}

    function renderRows(rows) {{
      if (!rows.length) {{
        library.innerHTML = '<p class="empty">No spectrum library rows yet.</p>';
        return;
      }}
      const rowHtml = row => {{
        const id = escapeHtml(row.analysis_id || '');
        const figs = (row.plot_urls || []).map(url => `<figure><img src="${{escapeHtml(url)}}" alt="Fit plot"></figure>`).join('');
        const detail = figs ? `<tr class="detail" data-detail="${{id}}" hidden><td colspan="9"><div class="plots">${{figs}}</div></td></tr>` : '';
        const plotsBtn = figs ? `<button type="button" class="link-btn" data-plots="${{id}}">Plots</button>` : '';
        return `
        <tr>
          <td>${{escapeHtml(row.collection_date)}}</td>
          <td>${{escapeHtml(row.collection_time)}}</td>
          <td>${{escapeHtml(row.comparison)}}</td>
          <td>${{Number(row.isotope_shift_MHz || 0).toFixed(3)}}</td>
          <td>${{Number(row.isotope_shift_total_unc_MHz || 0).toFixed(3)}}</td>
          <td>${{escapeHtml(row.fit_backend)}}</td>
          <td>${{escapeHtml(row.scans_removed || '0')}}</td>
          <td>${{escapeHtml(row.run_label)}}</td>
          <td class="row-actions">${{plotsBtn}}<button type="button" class="link-btn danger" data-delete="${{id}}">Delete</button></td>
        </tr>${{detail}}`;
      }};
      library.innerHTML = '<table><thead><tr><th>Date</th><th>Time</th><th>Comparison</th><th>Shift (MHz)</th><th>Unc. (MHz)</th><th>Fit</th><th>Bad Scans</th><th>Run</th><th>Actions</th></tr></thead><tbody>' + rows.map(rowHtml).join('') + '</tbody></table>';
    }}

    library.addEventListener('click', async event => {{
      const plotsBtn = event.target.closest('[data-plots]');
      if (plotsBtn) {{
        const detail = library.querySelector('[data-detail="' + plotsBtn.getAttribute('data-plots') + '"]');
        if (detail) detail.hidden = !detail.hidden;
        return;
      }}
      const delBtn = event.target.closest('[data-delete]');
      if (delBtn) {{
        const id = delBtn.getAttribute('data-delete');
        if (!confirm('Delete this library row? This also removes its saved fit plots.')) return;
        try {{
          const data = await withBusy(delBtn, () => postJson('/api/delete_row', {{ analysis_id: id }}));
          renderRows(data.library.rows);
          renderPlots(data.library.rows);
          renderSummaryPlots(data.library.summary_plot_urls || []);
          await refreshUncertainty();
          setStatus(`Deleted ${{data.deleted}} row(s).`, 'ok');
        }} catch (error) {{
          setStatus(error.message, 'err', error.detail);
        }}
      }}
    }});

    function renderPlots(rows) {{
      const seen = new Set();
      const urls = [];
      rows.forEach(row => {{
        (row.plot_urls || []).forEach(url => {{
          if (!seen.has(url) && urls.length < 4) {{
            seen.add(url);
            urls.push(url);
          }}
        }});
      }});
      plots.innerHTML = urls.length ? urls.map(url => `<figure><img src="${{url}}" alt="Saved spectrum fit plot"></figure>`).join('') : '<p class="empty">Run an analysis to preview saved fit plots here.</p>';
    }}

    function renderSummaryPlots(urls) {{
      summaryPlots.innerHTML = urls.length ? urls.map(url => `<figure><img src="${{url}}" alt="Library stability summary plot"></figure>`).join('') : '<p class="empty">Add library rows to build stability summary plots.</p>';
    }}

    function fmt(value, digits) {{
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(digits || 2) : 'n/a';
    }}

    function renderUncertainty(data) {{
      const freq = data.frequentist || {{}};
      const bayes = data.bayesian || {{}};
      uncSummary.innerHTML = `
        <div class="summary-card"><strong>${{data.included.length}}</strong><span>included rows</span></div>
        <div class="summary-card"><strong>${{data.excluded.length}}</strong><span>excluded rows</span></div>
        <div class="summary-card"><strong>${{fmt(freq.weighted_mean_MHz)}} +/- ${{fmt(freq.weighted_scatter_sem_MHz)}}</strong><span>weighted mean +/- scatter SEM MHz</span></div>
        <div class="summary-card"><strong>${{fmt(freq.chi2_red)}}</strong><span>reduced chi-squared</span></div>
        <div class="summary-card"><strong>${{bayes.available ? fmt(bayes.mu_mean_MHz) + ' +/- ' + fmt(bayes.mu_sd_MHz) : 'n/a'}}</strong><span>Bayesian shared shift MHz</span></div>
        <div class="summary-card"><strong>${{bayes.available ? fmt(bayes.sigma_extra_mean_MHz) : 'n/a'}}</strong><span>extra scatter MHz</span></div>
      `;
      uncPlot.innerHTML = data.plot_url ? `<figure><img src="${{data.plot_url}}" alt="Uncertainty propagation analysis plot"></figure>` : `<p class="empty">No uncertainty plot was generated. ${{data.plot_error || ''}}</p>`;
      const rowHtml = (row, state) => `
        <tr>
          <td>${{state}}</td>
          <td>${{row.collection_date || ''}}</td>
          <td>${{row.collection_time || ''}}</td>
          <td>${{row.run_label || ''}}</td>
          <td>${{fmt(row.shift_MHz, 3)}}</td>
          <td>${{fmt(row.fit_unc_MHz, 2)}}</td>
          <td>${{fmt(row.total_unc_MHz, 2)}}</td>
          <td>${{row.num_points_reference || 0}} / ${{row.num_points_comparison || 0}}</td>
          <td>${{(row.reasons || []).join('; ')}}</td>
        </tr>`;
      const allRows = data.included.map(row => rowHtml(row, 'kept')).concat(data.excluded.map(row => rowHtml(row, 'cut')));
      uncTable.innerHTML = allRows.length
        ? '<table><thead><tr><th>State</th><th>Date</th><th>Time</th><th>Run</th><th>Shift</th><th>Fit unc</th><th>Total unc</th><th>Points</th><th>Reason</th></tr></thead><tbody>' + allRows.join('') + '</tbody></table>'
        : '<p class="empty">No rows match this comparison.</p>';
    }}

    async function refreshUncertainty() {{
      const params = new URLSearchParams({{
        comparison: document.getElementById('unc-comparison').value,
        fit_unc_cut_MHz: document.getElementById('unc-fit-cut').value,
        total_unc_cut_MHz: document.getElementById('unc-total-cut').value,
        min_points_per_isotope: document.getElementById('unc-min-points').value,
        max_peak_to_model: document.getElementById('unc-peak-model').value,
        require_bracket_pass: document.getElementById('unc-bracket').checked ? '1' : '0'
      }});
      const response = await fetch('/api/uncertainty?' + params.toString());
      const data = await response.json();
      if (!response.ok) {{ const err = new Error(data.error || 'Uncertainty analysis failed'); err.detail = data.detail; throw err; }}
      renderUncertainty(data);
    }}

    async function refreshLibrary() {{
      const response = await fetch('/api/library');
      const data = await response.json();
      renderRows(data.rows);
      renderPlots(data.rows);
      renderSummaryPlots(data.summary_plot_urls || []);
      await refreshUncertainty();
    }}

    form.addEventListener('submit', async event => {{
      event.preventDefault();
      const submitBtn = form.querySelector('button[type="submit"]');
      const formData = new FormData(form);
      const payload = Object.fromEntries(formData.entries());
      const start = payload.collection_time_start || '';
      const end = payload.collection_time_end || '';
      payload.collection_time = (start && end) ? (start + '-' + end) : (start || end || '');
      try {{
        setStatus('Running fit and saving spectrum library rows...', '');
        const data = await withBusy(submitBtn, () => postJson('/api/analyze', payload));
        renderRows(data.library.rows);
        renderPlots(data.library.rows);
        renderSummaryPlots(data.library.summary_plot_urls || []);
        await refreshUncertainty();
        const rowsText = data.rows.map(row => `${{row.comparison}} ${{Number(row.isotope_shift_MHz).toFixed(3)}} +/- ${{Number(row.isotope_shift_total_unc_MHz).toFixed(3)}} MHz`).join('; ');
        const skipped = data.skipped_duplicates ? ` Skipped ${{data.skipped_duplicates}} duplicate row(s).` : '';
        const replaced = data.replaced_rows ? ` Replaced ${{data.replaced_rows}} existing row(s).` : '';
        const merged = data.merged_file_count ? ` Fit used ${{data.merged_file_count}} merged file(s).` : '';
        let rejected = '';
        if (data.rejected_details && data.rejected_details.length) {{
          rejected = ' Rejected ' + data.rejected_details.length + ' row(s): ' + data.rejected_details.map(d => `[${{d.comparison}}] ${{(d.reasons || []).join(', ')}}`).join(' | ');
        }}
        setStatus(`Saved ${{data.rows.length}} row(s). ${{rowsText}}${{replaced}}${{merged}}${{skipped}}${{rejected}}`, 'ok');
      }} catch (error) {{
        setStatus(error.message, 'err', error.detail);
      }}
    }});

    document.getElementById('refresh').addEventListener('click', event => {{
      withBusy(event.currentTarget, refreshLibrary).then(() => setStatus('Library refreshed.', 'ok')).catch(error => setStatus(error.message, 'err', error.detail));
    }});

    document.getElementById('unc-refresh').addEventListener('click', event => {{
      withBusy(event.currentTarget, refreshUncertainty).then(() => setStatus('Uncertainty analysis refreshed.', 'ok')).catch(error => setStatus(error.message, 'err', error.detail));
    }});

    document.getElementById('unc-export').addEventListener('click', () => {{
      const params = new URLSearchParams({{
        comparison: document.getElementById('unc-comparison').value,
        fit_unc_cut_MHz: document.getElementById('unc-fit-cut').value,
        total_unc_cut_MHz: document.getElementById('unc-total-cut').value,
        min_points_per_isotope: document.getElementById('unc-min-points').value,
        max_peak_to_model: document.getElementById('unc-peak-model').value,
        require_bracket_pass: document.getElementById('unc-bracket').checked ? '1' : '0'
      }});
      window.location = '/api/export?' + params.toString();
      setStatus('Exporting included set as CSV...', 'ok');
    }});

    document.getElementById('rebuild').addEventListener('click', async event => {{
      if (!confirm('Rebuild the library from the data folder? This replaces the current library rows.')) return;
      const rebuildBtn = event.currentTarget;
      const payload = {{
        data_dir: document.getElementById('data-dir').value,
        auto_remove_bad_scans: form.elements.auto_remove_bad_scans.checked ? 'on' : '',
        include_background: form.elements.include_background.checked ? 'on' : '',
        include_36s: form.elements.include_36s.checked ? 'on' : '',
        tof_gate_32S: document.getElementById('tof-gate-32').value,
        tof_gate_34S: document.getElementById('tof-gate-34').value,
        tof_gate_36S: document.getElementById('tof-gate-36').value,
        bin_width_MHz: document.getElementById('bin-width').value,
        beam_voltage_unc_V: document.getElementById('hv-unc').value
      }};
      try {{
        setStatus('Rebuilding from the selected data folder...', '');
        const data = await withBusy(rebuildBtn, () => postJson('/api/rebuild', payload));
        renderRows(data.library.rows);
        renderPlots(data.library.rows);
        renderSummaryPlots(data.library.summary_plot_urls || []);
        await refreshUncertainty();
        setStatus(`Rebuilt ${{data.runs}} run(s), saved ${{data.rows_added}} row(s).`, 'ok');
      }} catch (error) {{
        setStatus(error.message, 'err', error.detail);
      }}
    }});
    refreshLibrary().then(() => setStatus('Library loaded.', 'ok')).catch(error => setStatus(error.message, 'err', error.detail));
  </script>
</body>
</html>"""
    return page.encode("utf-8")


class SpectrumLibraryHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(render_page(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/library":
            summary = library_summary()
            summary["rows"] = [row_to_view(row) for row in summary["rows"]]
            summary["summary_plot_urls"] = [plot_url(path) for path in refresh_summary_plots()]
            self.send_json(summary)
            return
        if parsed.path == "/api/uncertainty":
            try:
                self.send_json(uncertainty_summary_from_query(parse_qs(parsed.query)))
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc), "detail": traceback.format_exc()}, status=400)
            return
        if parsed.path == "/api/export":
            try:
                query = parse_qs(parsed.query)
                comparison, cuts = inclusion_cuts_from_query(query)
                result = analyze_library_uncertainty(read_library_csv(LIBRARY_CSV), cuts)
                csv_text = build_export_csv(comparison, result)
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc), "detail": traceback.format_exc()}, status=400)
                return
            filename = f"{safe_plot_label(comparison)}_included_{int(time.time())}.csv"
            try:
                EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
                (EXPORTS_DIR / filename).write_text(csv_text, encoding="utf-8")
            except OSError:
                pass
            body = csv_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/plot":
            query = parse_qs(parsed.query)
            requested = Path(unquote(query.get("path", [""])[0])).resolve()
            try:
                if not (
                    _is_relative_to(requested, PLOT_DIR.resolve())
                    or _is_relative_to(requested, UNCERTAINTY_PLOT_DIR.resolve())
                ):
                    raise ValueError
            except ValueError:
                self.send_json({"error": "Plot path is outside the plot directory."}, status=403)
                return
            if not requested.exists():
                self.send_json({"error": "Plot not found."}, status=404)
                return
            content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
            self.send_bytes(requested.read_bytes(), content_type)
            return
        self.send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/analyze", "/api/rebuild", "/api/delete_row"):
            self.send_json({"error": "Not found."}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            if parsed.path == "/api/delete_row":
                analysis_id = str(payload.get("analysis_id", "")).strip()
                if not analysis_id:
                    raise ValueError("Missing analysis_id.")
                with MUTATION_LOCK:
                    existing = read_library_csv(LIBRARY_CSV)
                    target = [r for r in existing if str(r.get("analysis_id", "")) == analysis_id]
                    if not target:
                        raise ValueError(f"No library row with analysis_id {analysis_id}.")
                    remaining = [r for r in existing if str(r.get("analysis_id", "")) != analysis_id]
                    _write_library_csv(LIBRARY_CSV, remaining)
                    _write_library_jsonl(LIBRARY_JSONL, remaining)
                    delete_row_plots(target)
                summary = library_summary()
                summary["rows"] = [row_to_view(row) for row in summary["rows"]]
                summary["summary_plot_urls"] = [plot_url(path) for path in refresh_summary_plots()]
                self.send_json({"deleted": len(target), "library": summary})
                return

            require_satlas_backend()
            if parsed.path == "/api/rebuild":
                options = options_from_payload(payload)
                with MUTATION_LOCK:
                    result = rebuild_library_from_folder(
                        payload.get("data_dir") or DEFAULT_DATA_DIR,
                        include_background=bool(payload.get("include_background")),
                        include_36s=bool(payload.get("include_36s")),
                        options=options,
                    )
                summary = library_summary()
                summary["rows"] = [row_to_view(row) for row in summary["rows"]]
                summary["summary_plot_urls"] = [plot_url(path) for path in refresh_summary_plots()]
                self.send_json(
                    {
                        "runs": len(result["runs"]),
                        "rows_added": len(result["rows"]),
                        "library": summary,
                    }
                )
                return

            files = split_file_input(payload.get("files", ""))
            if not files:
                raise ValueError("Paste at least two spectrum filenames.")

            options = options_from_payload(payload)
            explicit_labels = parse_isotope_labels(payload.get("isotopes"))
            collection_date = payload.get("collection_date") or ""
            collection_time = payload.get("collection_time") or ""
            run_label = payload.get("run_label") or ""
            replaced_rows: list[dict] = []
            analysis_files: list[str | Path] = files
            analysis_labels = explicit_labels
            if bool(payload.get("update_existing_day")):
                analysis_files, analysis_labels, replaced_rows = merge_with_existing_day_run(
                    files=files,
                    isotope_labels=explicit_labels,
                    data_dir=payload.get("data_dir") or DEFAULT_DATA_DIR,
                    collection_date=collection_date,
                    collection_time=collection_time,
                    run_label=run_label,
                    options=options,
                )
            if not replaced_rows and len(set(analysis_labels or [])) < 2:
                raise ValueError(
                    "Only one isotope was found in the pasted files. To add scans to an existing day, "
                    "leave 'Update matching day/run entry' checked and use the same run label as the saved row."
                )

            rows = run_analysis(
                analysis_files,
                data_dir=payload.get("data_dir") or DEFAULT_DATA_DIR,
                isotope_labels=analysis_labels,
                collection_date=collection_date,
                collection_time=collection_time,
                run_label=run_label,
                transition=payload.get("transition") or "",
                notes=payload.get("notes") or "",
                options=options,
                plot_dir=PLOT_DIR,
                library_csv=None,
                library_jsonl=None,
                adjacent_single_scan_pairs=True,
            )
            quality_min_points = int(
                float(str(payload.get("quality_min_points") or MIN_LIBRARY_POINTS_PER_ISOTOPE))
            )
            quality_max_peak = float(
                str(payload.get("quality_max_peak_to_model") or MAX_LIBRARY_PEAK_TO_MODEL)
            )
            kept_rows: list[dict] = []
            rejected_rows: list[dict] = []
            rejected_details: list[dict] = []
            for row in rows:
                reasons = library_quality_reasons(
                    row, min_points=quality_min_points, max_peak_to_model=quality_max_peak
                )
                if reasons:
                    rejected_rows.append(row)
                    rejected_details.append(
                        {
                            "comparison": row.get("comparison", ""),
                            "isotope_shift_MHz": row.get("isotope_shift_MHz", ""),
                            "run_label": row.get("run_label", ""),
                            "reasons": reasons,
                        }
                    )
                else:
                    kept_rows.append(row)
            rows = kept_rows
            if rejected_rows:
                delete_row_plots(rejected_rows)
            if not rows:
                detail = " ".join(
                    f"[{d['comparison']}] {'; '.join(d['reasons'])}" for d in rejected_details
                )
                raise ValueError(
                    "No rows passed the library quality gates. " + detail
                )
            with MUTATION_LOCK:
                if replaced_rows:
                    replace_library_group(replaced_rows, rows)
                    saved_rows = rows
                    skipped_duplicates = 0
                else:
                    saved_rows = append_new_library_rows_csv(LIBRARY_CSV, rows)
                    if saved_rows:
                        append_library_jsonl(LIBRARY_JSONL, saved_rows)
                    skipped_duplicates = len(rows) - len(saved_rows)
            summary = library_summary()
            summary["rows"] = [row_to_view(row) for row in summary["rows"]]
            summary["summary_plot_urls"] = [plot_url(path) for path in refresh_summary_plots()]
            self.send_json(
                {
                    "rows": [row_to_view(row) for row in saved_rows],
                    "skipped_duplicates": skipped_duplicates,
                    "replaced_rows": len(replaced_rows),
                    "merged_file_count": len(analysis_files) if replaced_rows else 0,
                    "rejected_rows": len(rejected_rows),
                    "rejected_details": rejected_details,
                    "library": summary,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc), "detail": traceback.format_exc()}, status=400)


def run_server(host: str = "127.0.0.1", port: int = 8766, open_browser: bool = True) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), SpectrumLibraryHandler)
    url = f"http://{host}:{port}"
    if sys.stdout is not None:
        print(f"CREMA Spectrum Library GUI: {url}")
        print(f"Library CSV: {LIBRARY_CSV}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CREMA spectrum library GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        (ROOT / "spectrum_library_gui_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
