"""Re-derive the beam-energy library rows with the GUI's v2 rest-frame method (tail model).

Backs up energy_correction_library.{csv,jsonl}, recomputes the five June 17/18
collinear/anticollinear pairs with compute_beam_energy_correction_v2 (the same code the
Beam-energy tab now runs for v2 options), replaces the rows with the same file sets, and
prints the global offset and the beam-energy systematic on 34S-32S.

    ..\\.venv\\Scripts\\python.exe regenerate_energy_library.py
"""
import csv, json, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "hfs_gui"))
OVERRIDES = {"tail_model": "exponential", "shape_transfer": "tail", "voltage_offset_V": 190.77172150007803}


def run_pair(args):
    import matplotlib; matplotlib.use("Agg")
    import spectrum_library_gui as gui
    cols, antis, notes = args
    options = gui.load_default_options(); options.update(OVERRIDES)
    t0 = time.time()
    result = gui.compute_beam_energy_correction(cols, antis, "32S", options, gui.DEFAULT_DATA_DIR)
    label = "v2 rest frame, exponential energy-loss tail, per-bunch HV (2026-09-29)"
    row = gui.energy_result_to_row(result, cols, antis, f"{label}; {notes}" if notes else label)
    return row, result, time.time() - t0


if __name__ == "__main__":
    import spectrum_library_gui as gui
    old = gui.read_energy_library()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (gui.ENERGY_LIBRARY_CSV, gui.ENERGY_LIBRARY_JSONL):
        backup = path.with_name(f"{path.stem}_before_v2tail_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        print("backup", backup.name)
    jobs = [([f for f in r["collinear_files"].split(";") if f], [f for f in r["anticollinear_files"].split(";") if f], r.get("notes", ""))
            for r in old]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        outs = list(pool.map(run_pair, jobs))
    for (row, result, dt), r_old in zip(outs, old):
        gui.append_energy_row(row)
        print(f"{row['collinear_files']:52s} | {row['anticollinear_files']:52s} | offset {float(r_old['delta_V']):7.2f} -> "
              f"{result['delta_V']:7.2f} +/- {result['voltage_inferred_unc_V']:.2f} V  ({dt:.0f}s)")
    glob = gui.compute_global_energy_average()
    sys_info = gui.beam_energy_systematic("34S-32S")
    print(f"\nglobal offset (cluster mean of delta_V): {glob['delta_V']:.4f} +/- {glob['delta_V_unc_V']:.4f} V "
          f"({glob['n_clusters']} clusters, {glob['n_rows']} rows)")
    print(f"beam-energy systematic 34S-32S: sigma_V {sys_info['sigma_V']:.3f} V x {sys_info['d_is_dv_MHz_per_V']:.3f} MHz/V "
          f"= {sys_info['beam_sys_MHz']:.3f} MHz")
    json.dump({"global": glob, "systematic": sys_info}, open(Path(__file__).with_name("energy_library_v2tail.json"), "w"),
              indent=1, default=float)
