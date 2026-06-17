"""Sweep QMF DC voltage and plot isotope transmission curves.

This is intended for fixed-hardware scans where RF amplitude/frequency and
source geometry are held constant while Vdc is swept. Transmission is counted
from QMF.lua's trap event log as ions that reach ``stop_y_mm``.
"""

from __future__ import annotations

import argparse
import csv
import math
import uuid
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


QMF_DIR = Path(r"C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF")
SIMION = Path(r"C:\Program Files\SIMION-8.1\simion.exe")
IOB = QMF_DIR / "QMF.iob"
EVENT_CSV = QMF_DIR / "data" / "trap_events.csv"
TOB_MIN_USEC = 0.0
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


@dataclass(frozen=True)
class Setting:
    scenario: str
    mass: int
    rf_freq_mhz: float
    qmf_rf: float
    qmf_dc: float
    brubaker_rf: float
    source_radius_mm: float
    half_angle_deg: float
    mean_ke_ev: float
    fwhm_ke_ev: float


def parse_masses(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_vdcs(text: str | None, vdc_min: float, vdc_max: float, vdc_step: float) -> list[float]:
    if text:
        return [float(part.strip()) for part in text.split(",") if part.strip()]
    count = int(math.floor((vdc_max - vdc_min) / vdc_step + 0.5)) + 1
    return [round(vdc_min + i * vdc_step, 10) for i in range(count)]


def write_fly2(setting: Setting, num_particles: int) -> Path:
    data_dir = QMF_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    temp_fly2 = (
        data_dir
        / (
            f"_vdc_sweep_{setting.scenario}_m{setting.mass}_"
            f"dc{setting.qmf_dc:.3f}_ke{setting.mean_ke_ev:.3f}_"
            f"f{setting.fwhm_ke_ev:.3f}_{uuid.uuid4().hex[:8]}.fly2"
        )
    )
    if setting.fwhm_ke_ev <= 0:
        ke_block = f"ke = {setting.mean_ke_ev}"
    else:
        ke_block = (
            "ke = gaussian_distribution {\n"
            f"      mean = {setting.mean_ke_ev},\n"
            f"      fwhm = {setting.fwhm_ke_ev}\n"
            "    }"
        )
    temp_fly2.write_text(
        f"""particles {{
  coordinates = 0,
  standard_beam {{
    n = {num_particles},
    tob = uniform_distribution {{
      min = {TOB_MIN_USEC},
      max = {1.0 / setting.rf_freq_mhz}
    }},
    mass = {setting.mass},
    charge = 1,
    cwf = 1,
    color = 0,
    direction = cone_direction_distribution {{
      axis = vector(0, 1, 0),
      half_angle = {setting.half_angle_deg},
      fill = true
    }},
    position = circle_distribution {{
      center = vector(19, 19, 19),
      normal = vector(0, 1, 0),
      radius = {setting.source_radius_mm},
      fill = true
    }},
    {ke_block}
  }}
}}
""",
        encoding="ascii",
    )
    return temp_fly2


def run_simion_once(setting: Setting, args: argparse.Namespace, n_particles: int) -> None:
    temp_fly2 = write_fly2(setting, n_particles)
    cmd = [
        str(SIMION),
        "--nogui",
        "--noprompt",
        "fly",
        "--trajectory-quality",
        str(args.trajectory_quality),
        "--particles",
        str(temp_fly2),
        "--grouped",
        "1",
        "--programs",
        "1",
        "--retain-trajectories",
        "0",
        "--adjustable",
        f"RF_freq_MHz={setting.rf_freq_mhz}",
        "--adjustable",
        f"QMF_RF_amp_Vp={setting.qmf_rf}",
        "--adjustable",
        f"QMF_DC_V={setting.qmf_dc}",
        "--adjustable",
        f"entrance_Brubaker_RF_amp_Vp={setting.brubaker_rf}",
        "--adjustable",
        f"exit_Brubaker_RF_amp_Vp={setting.brubaker_rf}",
        "--adjustable",
        f"max_flight_time_us={args.max_flight_time_us}",
        "--adjustable",
        f"stop_y_mm={args.stop_y_mm}",
        "--adjustable",
        f"min_x_mm={args.min_x_mm}",
        "--adjustable",
        f"max_x_mm={args.max_x_mm}",
        "--adjustable",
        f"min_z_mm={args.min_z_mm}",
        "--adjustable",
        f"max_z_mm={args.max_z_mm}",
        "--adjustable",
        f"min_y_mm={args.min_y_mm}",
    ]
    if args.pressure_pa is not None and args.pressure_pa >= 0:
        cmd += ["--adjustable", f"pressure_pa={args.pressure_pa}"]
    cmd.append(str(IOB))
    subprocess.run(
        cmd,
        cwd=QMF_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def run_simion(setting: Setting, args: argparse.Namespace, n_particles: int) -> None:
    for attempt in range(args.retries + 1):
        try:
            run_simion_once(setting, args, n_particles)
            return
        except subprocess.CalledProcessError:
            if attempt == args.retries:
                raise
            time.sleep(args.retry_delay_s)


def run_point(setting: Setting, args: argparse.Namespace) -> dict[str, int]:
    """Fly num_particles in chunks (SIMION scales badly past ~25-50/run) and sum events."""
    chunk = args.chunk_size if args.chunk_size and args.chunk_size > 0 else args.num_particles
    total = {"timeout_trapped": 0, "reached_stop_plane": 0, "out_of_bounds": 0}
    remaining = args.num_particles
    while remaining > 0:
        n = min(chunk, remaining)
        run_simion(setting, args, n)  # QMF.lua rewrites trap_events.csv each fly
        counts = count_events()
        for k in total:
            total[k] += counts[k]
        remaining -= n
    return total


def count_events() -> dict[str, int]:
    counts = {"timeout_trapped": 0, "reached_stop_plane": 0, "out_of_bounds": 0}
    if not EVENT_CSV.exists():
        return counts
    seen: set[str] = set()
    with EVENT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ion = row.get("ion")
            event = row.get("event")
            if not ion or ion in seen:
                continue
            seen.add(ion)
            if event in counts:
                counts[event] += 1
    return counts


def row_key(row: dict[str, str]) -> tuple[str, float, int]:
    return (row["scenario"], round(float(row["qmf_dc"]), 10), int(row["mass"]))


def load_existing(output: Path) -> dict[tuple[str, float, int], dict[str, str]]:
    if not output.exists():
        return {}
    with output.open(newline="", encoding="utf-8") as f:
        return {row_key(row): row for row in csv.DictReader(f)}


def append_row(output: Path, row: dict) -> None:
    output.parent.mkdir(exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _plot_metric(
    rows: list[dict],
    plot_dir: Path,
    metric: str,
    ylabel: str,
    file_stem: str,
    title_prefix: str,
    title_suffix: str,
) -> None:
    scenarios = sorted({row["scenario"] for row in rows})
    masses = sorted({row["mass"] for row in rows})
    colors = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for mass in masses:
            points = sorted([row for row in subset if row["mass"] == mass], key=lambda r: r["qmf_dc"])
            if not points:
                continue
            ax.plot(
                [row["qmf_dc"] for row in points],
                [row[metric] for row in points],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=f"{mass}S",
                color=colors.get(mass),
            )
        ax.set_xlabel("QMF DC voltage (V)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Mass")
        ax.set_title(f"{title_prefix} vs Vdc ({scenario.replace('_', ' ')}){title_suffix}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{file_stem}_{scenario}.png")
        plt.close(fig)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 4.2 * len(scenarios)), dpi=160, sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        subset = [row for row in rows if row["scenario"] == scenario]
        for mass in masses:
            points = sorted([row for row in subset if row["mass"] == mass], key=lambda r: r["qmf_dc"])
            ax.plot(
                [row["qmf_dc"] for row in points],
                [row[metric] for row in points],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=f"{mass}S",
                color=colors.get(mass),
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Mass")
        ax.set_title(scenario.replace("_", " "))
    axes[-1].set_xlabel("QMF DC voltage (V)")
    fig.suptitle(f"{title_prefix} vs Vdc{title_suffix}", y=0.995)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{file_stem}_compare_spreads.png")
    plt.close(fig)


def plot_results(output: Path, plot_dir: Path, title_suffix: str) -> None:
    if not output.exists():
        return
    rows = []
    with output.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["qmf_dc"] = float(row["qmf_dc"])
            row["mass"] = int(row["mass"])
            row["transmission_fraction"] = float(row["transmission_fraction"])
            row["trapped_fraction"] = float(row["trapped_fraction"])
            rows.append(row)
    if not rows:
        return
    plot_dir.mkdir(exist_ok=True)
    _plot_metric(
        rows,
        plot_dir,
        "transmission_fraction",
        "Transmission fraction",
        "vdc_transmission",
        "QMF transmission",
        title_suffix,
    )
    _plot_metric(
        rows,
        plot_dir,
        "trapped_fraction",
        "Trapped fraction",
        "vdc_trapping",
        "QMF trapping",
        title_suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masses", default="32,34,36")
    parser.add_argument("--num-particles", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="particles per SIMION fly; <=0 flies all at once (slow >~50)")
    parser.add_argument("--pressure-pa", type=float, default=None,
                        help="override QMF.lua pressure_pa (e.g. 0 for collisionless); None = lua default")
    parser.add_argument("--trajectory-quality", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-s", type=float, default=2.0)
    parser.add_argument("--rf-freq-mhz", type=float, default=2.4)
    parser.add_argument("--qmf-rf", type=float, default=448.0)
    parser.add_argument("--brubaker-rf", type=float, default=448.0)
    parser.add_argument("--mean-ke-ev", type=float, default=0.7367136539184808)
    parser.add_argument("--realistic-fwhm-ke-ev", type=float, default=0.11451862433053576)
    parser.add_argument("--source-radius-mm", type=float, default=0.03)
    parser.add_argument("--half-angle-deg", type=float, default=0.10)
    parser.add_argument("--vdc-min", type=float, default=34.0)
    parser.add_argument("--vdc-max", type=float, default=44.0)
    parser.add_argument("--vdc-step", type=float, default=0.5)
    parser.add_argument("--vdcs", default=None)
    parser.add_argument("--max-flight-time-us", type=float, default=150.0)
    parser.add_argument("--stop-y-mm", type=float, default=170.0)
    parser.add_argument("--min-x-mm", type=float, default=12.0)
    parser.add_argument("--max-x-mm", type=float, default=26.0)
    parser.add_argument("--min-z-mm", type=float, default=12.0)
    parser.add_argument("--max-z-mm", type=float, default=26.0)
    parser.add_argument("--min-y-mm", type=float, default=10.0)
    parser.add_argument("--output", default=str(QMF_DIR / "data" / "vdc_transmission_sweep_fixed_rf448.csv"))
    parser.add_argument("--plot-dir", default=str(QMF_DIR / "data" / "plots"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    plot_dir = Path(args.plot_dir)
    if args.plot_only:
        plot_results(output, plot_dir, f" at {args.rf_freq_mhz:g} MHz, RF {args.qmf_rf:g} Vp")
        return

    masses = parse_masses(args.masses)
    vdcs = parse_vdcs(args.vdcs, args.vdc_min, args.vdc_max, args.vdc_step)
    scenarios = [
        ("monoenergetic", 0.0),
        ("realistic_spread", args.realistic_fwhm_ke_ev),
    ]
    existing = load_existing(output)
    total = len(scenarios) * len(vdcs) * len(masses)
    completed = len(existing)

    for scenario, fwhm in scenarios:
        for vdc in vdcs:
            for mass in masses:
                key = (scenario, round(float(vdc), 10), mass)
                if key in existing:
                    continue
                setting = Setting(
                    scenario=scenario,
                    mass=mass,
                    rf_freq_mhz=args.rf_freq_mhz,
                    qmf_rf=args.qmf_rf,
                    qmf_dc=vdc,
                    brubaker_rf=args.brubaker_rf,
                    source_radius_mm=args.source_radius_mm,
                    half_angle_deg=args.half_angle_deg,
                    mean_ke_ev=args.mean_ke_ev,
                    fwhm_ke_ev=fwhm,
                )
                counts = run_point(setting, args)
                completed += 1
                row = {
                    "scenario": scenario,
                    "mass": mass,
                    "qmf_dc": vdc,
                    "dc_rf_ratio": vdc / args.qmf_rf,
                    "rf_freq_mhz": args.rf_freq_mhz,
                    "qmf_rf": args.qmf_rf,
                    "brubaker_rf": args.brubaker_rf,
                    "mean_ke_ev": args.mean_ke_ev,
                    "fwhm_ke_ev": fwhm,
                    "source_radius_mm": args.source_radius_mm,
                    "half_angle_deg": args.half_angle_deg,
                    "num_particles": args.num_particles,
                    "max_flight_time_us": args.max_flight_time_us,
                    "stop_y_mm": args.stop_y_mm,
                    "min_x_mm": args.min_x_mm,
                    "max_x_mm": args.max_x_mm,
                    "min_z_mm": args.min_z_mm,
                    "max_z_mm": args.max_z_mm,
                    "min_y_mm": args.min_y_mm,
                    "transmitted": counts["reached_stop_plane"],
                    "trapped": counts["timeout_trapped"],
                    "out_of_bounds": counts["out_of_bounds"],
                    "transmission_fraction": counts["reached_stop_plane"] / args.num_particles,
                    "trapped_fraction": counts["timeout_trapped"] / args.num_particles,
                    "out_of_bounds_fraction": counts["out_of_bounds"] / args.num_particles,
                }
                append_row(output, row)
                print(
                    f"{completed}/{total} {scenario} Vdc={vdc:.3f} m={mass}: "
                    f"T={row['transmitted']}/{args.num_particles} "
                    f"trap={row['trapped']}/{args.num_particles} "
                    f"oob={row['out_of_bounds']}/{args.num_particles}",
                    flush=True,
                )

    plot_results(output, plot_dir, f" at {args.rf_freq_mhz:g} MHz, RF {args.qmf_rf:g} Vp")
    print(f"Wrote {output}")
    print(f"Wrote plots in {plot_dir}")


if __name__ == "__main__":
    main()
