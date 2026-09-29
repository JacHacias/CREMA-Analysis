"""Replay the isotope-shift library from raw scans under a sequence of analysis settings.

Each library row is re-fitted from its own files and stored options, with one
settings step layered on top. The default sequence adds the v2 corrections one
at a time, so the change in every row and in the library weighted mean
(library_uncertainty_analysis, default cuts) can be attributed:

  legacy        stored rows (satlas2/curve_fit Voigt on raw binned counts)
  poisson       v2 Poisson ML on the same raw binned counts, plain Voigt
  +exposure     fit ions/bunch against the bunch exposure (dwell groups)
  +echo         remove MagneTOF echo counts
  +deadtime     dead-time live-time correction
  +ripple       60 Hz ripple kernel, 4.5 V amplitude
  +laser        15*sqrt(2) MHz laser linewidth  (= full v2)

Checks run on the full v2 settings: ripple amplitude free, laser as a
Lorentzian, and legacy+echo (echo removal alone in the old fitter).

Nothing in the live library is modified. Outputs go to --out (CSV of every
row x step, JSON summary); --write-library also saves the full-v2 rows as a
library pair (csv + jsonl) at the given path stem.

    .venv\\Scripts\\python.exe reanalyze_library.py --out analysis_plots\\reanalysis_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import library_uncertainty_analysis as lua
import quick_isotope_shift as qis

REPO = Path(__file__).resolve().parent
LIBRARY = REPO / "hfs_gui" / "data_library" / "isotope_shift_library.csv"
MOVED_DATA = {
    r"C:\Users\EMALAB\Documents\Jackson\S data for analysis":
        r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis",
}

OFF = {"remove_echo_counts": False, "deadtime_correction": False, "exposure_normalization": False,
       "ripple_amplitude_V": 0.0, "laser_linewidth_fwhm_MHz": 0.0}
V2_BOOT = dict(qis.rate_fit.V2_DEFAULTS)
V2 = {**V2_BOOT, "bootstrap_replicas": 0}  # the ablation steps compare centers only
STEPS: list[tuple[str, dict[str, Any] | None]] = [
    ("legacy", None),
    ("poisson", {**V2, **OFF}),
    ("+exposure", {**V2, **OFF, "exposure_normalization": True}),
    ("+echo", {**V2, **OFF, "exposure_normalization": True, "remove_echo_counts": True}),
    ("+deadtime", {**V2, **OFF, "exposure_normalization": True, "remove_echo_counts": True, "deadtime_correction": True}),
    ("+ripple", {**V2, "laser_linewidth_fwhm_MHz": 0.0}),
    ("+laser", dict(V2)),
    ("v2+bootstrap", dict(V2_BOOT)),
    ("v2_ripple_free", {**V2, "fit_ripple_amplitude": True}),
    ("v2_laser_lorentzian", {**V2, "laser_lineshape": "lorentzian"}),
    ("legacy+echo", {"analysis_model": "legacy", "remove_echo_counts": True}),
    # Adopted 2026-09-29: exponential energy-loss tail, 32S tail carried into the 34S fit,
    # pair bootstrap; run with --voltage-offset from the v2 tail-model calibration.
    ("v2+tail", {**V2_BOOT, "tail_model": "exponential", "shape_transfer": "tail"}),
]


def _resolve(path: str) -> Path:
    for old, new in MOVED_DATA.items():
        if path.startswith(old):
            return Path(new + path[len(old):])
    return Path(path)


def _labels(files: list[Path], options: dict[str, Any]) -> list[str]:
    labels = []
    for path in files:
        try:
            labels.append(qis.infer_isotope_label(path))
            continue
        except ValueError:
            pass
        wn = pd.read_csv(path, usecols=[options.get("wn_col", "wavemeter_wn1")]).iloc[:, 0]
        median = float(np.nanmedian(wn.to_numpy(dtype=float)))
        windows = options.get("isotope_wavenumber_windows") or qis.DEFAULT_ANALYSIS_OPTIONS["isotope_wavenumber_windows"]
        labels.append(next(k for k, (lo, hi) in windows.items() if lo <= median <= hi))
    return labels


def replay_row(row: dict[str, Any], overrides: dict[str, Any], plot_dir: Path) -> dict[str, Any]:
    """Re-fit one library row with its stored options updated by ``overrides``."""
    options = json.loads(row["options_json"])
    options.update(overrides)
    files = [_resolve(f) for f in row["files"].split(";") if f]
    labels = _labels(files, options)
    common = dict(collection_date=row["collection_date"], collection_time=row.get("collection_time", ""),
                  run_label=row["run_label"], transition=row.get("transition", ""), notes=row.get("notes", ""),
                  options=options, plot_dir=plot_dir)
    if set(labels) == {"32S", "36S"} and len(files) == 2:
        blocks = qis.build_single_file_blocks(files, labels)
        new_rows = qis.run_adjacent_block_analyses(blocks, **common)
    else:
        new_rows = qis.run_analysis(files, isotope_labels=labels, library_csv=None, library_jsonl=None,
                                    adjacent_single_scan_pairs=True, **common)
    match = [r for r in new_rows if r["comparison"] == row["comparison"]]
    if not match:
        raise ValueError(f"replay produced no {row['comparison']} row")
    new = match[0]
    # Keep the stored identity so cuts (background/boundary labels) and grouping match.
    for key in ("analysis_id", "collection_date", "collection_time", "run_label", "notes"):
        new[key] = row.get(key, "")
    return new


def _quality_summary(new_row: dict[str, Any]) -> dict[str, Any]:
    try:
        quality = json.loads(new_row.get("bad_scan_filter", "") or "{}").get("fit_quality", {})
    except json.JSONDecodeError:
        return {}
    out = {}
    for label, q in quality.items():
        if not isinstance(q, dict) or "reduced_chi2" not in q:
            continue
        out[label] = {k: q.get(k) for k in ("reduced_chi2", "echo_fraction", "min_live_fraction",
                                             "deadtime_count_correction", "ripple_halfwidth_MHz",
                                             "ripple_halfwidth_unc_MHz", "sigma_doppler_MHz", "gamma_MHz")
                      if k in q}
    return out


def library_mean(rows: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    result = lua.analyze_library_uncertainty(rows, lua.InclusionCuts(comparison=comparison))
    freq = result.get("frequentist") or {}
    return {
        "N": freq.get("N", 0),
        "weighted_mean_MHz": freq.get("weighted_mean_MHz"),
        "scatter_sem_MHz": freq.get("weighted_scatter_sem_MHz"),
        "internal_unc_MHz": freq.get("internal_unc_MHz"),
        "weighted_std_MHz": freq.get("weighted_std_MHz"),
        "chi2_red": freq.get("chi2_red"),
        "included": [f"{v['collection_date']} {v['run_label']}" for v in result.get("included", [])],
        "excluded": [f"{v['collection_date']} {v['run_label']}: {'; '.join(v.get('reasons', []))}"
                     for v in result.get("excluded", [])],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default=str(LIBRARY))
    parser.add_argument("--out", default=str(REPO / "analysis_plots" / "reanalysis_v2"))
    parser.add_argument("--steps", default="", help="comma-separated subset of step names")
    parser.add_argument("--write-library", default="", help="path stem for the full-v2 library (csv+jsonl)")
    parser.add_argument("--plot-dir", default="", help="keep fit plots here (default: a temp dir)")
    parser.add_argument("--jobs", type=int, default=1, help="rows fitted in parallel (worker processes)")
    parser.add_argument("--library-step", default="v2+bootstrap", help="step whose rows --write-library saves")
    parser.add_argument("--voltage-offset", type=float, default=None,
                        help="beam-energy offset (V) applied to every v2 step (default: each row's stored value)")
    parser.add_argument("--notes-replace", default="", help="OLD=>NEW substitution in the notes of the written rows")
    args = parser.parse_args(argv)

    stored = list(csv.DictReader(open(args.library, newline="", encoding="utf-8")))
    wanted = [s.strip() for s in args.steps.split(",") if s.strip()]
    steps = [(name, over) for name, over in STEPS if not wanted or name in wanted]
    if args.voltage_offset is not None:
        steps = [(name, over if over is None or over.get("analysis_model") == "legacy"
                  else {**over, "voltage_offset_V": args.voltage_offset}) for name, over in steps]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_root = Path(args.plot_dir) if args.plot_dir else Path(tempfile.mkdtemp(prefix="reanalysis_plots_"))

    table: list[dict[str, Any]] = []
    rows_by_step: dict[str, list[dict[str, Any]]] = {}
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    for name, overrides in steps:
        step_rows = []
        t0 = time.time()
        plot_dir = plot_root / name.replace("+", "p")
        if overrides is None:
            outcomes = [dict(row) for row in stored]
        elif pool is not None:
            futures = [pool.submit(replay_row, row, overrides, plot_dir) for row in stored]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except Exception as exc:  # report and keep going
                    outcomes.append(exc)
        else:
            outcomes = []
            for row in stored:
                try:
                    outcomes.append(replay_row(row, overrides, plot_dir))
                except Exception as exc:  # report and keep going
                    outcomes.append(exc)
        for index, (row, new) in enumerate(zip(stored, outcomes)):
            if isinstance(new, Exception):
                print(f"[{name}] row {index} {row['collection_date']} {row['comparison']}: FAILED {new}", flush=True)
                continue
            step_rows.append(new)
            table.append({
                "step": name,
                "row": index,
                "collection_date": row["collection_date"],
                "run_label": row["run_label"],
                "comparison": row["comparison"],
                "stored_IS_MHz": float(row["isotope_shift_MHz"]),
                "IS_MHz": float(new["isotope_shift_MHz"]),
                "IS_unc_MHz": float(new["isotope_shift_total_unc_MHz"]),
                "delta_vs_stored_MHz": float(new["isotope_shift_MHz"]) - float(row["isotope_shift_MHz"]),
                "quality": json.dumps(_quality_summary(new)),
            })
        rows_by_step[name] = step_rows
        means = {c: library_mean(step_rows, c) for c in ("34S-32S", "36S-32S")}
        m = means["34S-32S"]
        print(f"[{name}] {time.time() - t0:5.0f}s  34S-32S = {m['weighted_mean_MHz']:.2f} +/- {m['scatter_sem_MHz']:.2f} MHz "
              f"(N={m['N']}, chi2r={m['chi2_red']:.2f})  36S-32S = {means['36S-32S']['weighted_mean_MHz']}", flush=True)

    with open(out_dir / "reanalysis_rows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {name: {c: library_mean(rows, c) for c in ("34S-32S", "36S-32S")} for name, rows in rows_by_step.items()}
    (out_dir / "reanalysis_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    if pool is not None:
        pool.shutdown()
    if args.write_library and args.library_step in rows_by_step:
        stem = Path(args.write_library)
        stem.parent.mkdir(parents=True, exist_ok=True)
        rows = rows_by_step[args.library_step]
        if "=>" in args.notes_replace:
            before, after = args.notes_replace.split("=>", 1)
            for row in rows:
                row["notes"] = str(row.get("notes", "")).replace(before, after)
        with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=qis.LIBRARY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({k: row.get(k, "") for k in qis.LIBRARY_COLUMNS}, default=float) + "\n")
        print(f"wrote {stem.with_suffix('.csv')}")
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
