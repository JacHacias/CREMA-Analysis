"""Abundance-weighted view of the QMF Vdc trapping sweep.

Reads the trapping sweep CSV and produces, per scenario:
  1. Abundance-weighted trapped yield per isotope (log y) = trapped_fraction * natural abundance.
  2. Trapped-beam isotopic composition (purity) = weighted yield normalized to 100% at each Vdc.

Natural sulfur abundances (CIAAW): 32S 94.99%, 33S 0.75%, 34S 4.25%, 36S 0.01%.
33S is not simulated; the three modelled isotopes are renormalized among themselves
only for the purity panel (32/34/36), which is noted on the plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ABUNDANCE = {32: 0.9499, 34: 0.0425, 36: 0.0001}  # natural fraction
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]


def load(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "scenario": r["scenario"],
                    "mass": int(r["mass"]),
                    "qmf_dc": float(r["qmf_dc"]),
                    "trapped_fraction": float(r["trapped_fraction"]),
                }
            )
    return rows


def series(rows, scenario, mass):
    pts = sorted(
        [r for r in rows if r["scenario"] == scenario and r["mass"] == mass],
        key=lambda r: r["qmf_dc"],
    )
    v = np.array([p["qmf_dc"] for p in pts])
    tf = np.array([p["trapped_fraction"] for p in pts])
    return v, tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plot-dir", required=True)
    ap.add_argument("--title-suffix", default=" at 2.4 MHz, RF 448 Vp")
    args = ap.parse_args()

    rows = load(Path(args.csv))
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in ("monoenergetic", "realistic_spread") if any(r["scenario"] == s for r in rows)]

    # ---- Figure 1: abundance-weighted trapped yield (log y), one panel per scenario ----
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 4.4 * len(scenarios)), dpi=160, sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        for mass in MASSES:
            v, tf = series(rows, scenario, mass)
            if v.size == 0:
                continue
            w = tf * ABUNDANCE[mass]
            w = np.where(w > 0, w, np.nan)  # break lines at zero for log axis
            ax.plot(v, w, marker="o", ms=4, lw=1.8, color=COLORS[mass],
                    label=f"{mass}S ({ABUNDANCE[mass]*100:g}%)")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 2.0)
        ax.set_ylabel("Abundance-weighted\ntrapped fraction")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(title="Mass (nat. abund.)", fontsize=8)
        ax.set_title(scenario.replace("_", " "))
    axes[-1].set_xlabel("QMF DC voltage (V)")
    fig.suptitle(f"Abundance-weighted QMF trapping vs Vdc{args.title_suffix}", y=0.995)
    fig.tight_layout()
    out1 = plot_dir / "vdc_trapping_abundance_weighted.png"
    fig.savefig(out1)
    plt.close(fig)

    # ---- Figure 2: trapped-beam composition / purity (linear, stacked area) ----
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 4.4 * len(scenarios)), dpi=160, sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        # build common Vdc grid from 32S series (all masses share grid)
        vref, _ = series(rows, scenario, 32)
        comp = {}
        for mass in MASSES:
            v, tf = series(rows, scenario, mass)
            comp[mass] = tf * ABUNDANCE[mass]
        total = sum(comp[m] for m in MASSES)
        frac = {m: np.where(total > 0, comp[m] / total, 0.0) for m in MASSES}
        ax.stackplot(
            vref,
            [frac[m] for m in MASSES],
            colors=[COLORS[m] for m in MASSES],
            labels=[f"{m}S" for m in MASSES],
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Trapped-beam\ncomposition")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="center left", fontsize=8)
        ax.set_title(f"{scenario.replace('_', ' ')} (natural-abundance beam; 32/34/36 only)")
    axes[-1].set_xlabel("QMF DC voltage (V)")
    fig.suptitle(f"Trapped-beam isotopic purity vs Vdc{args.title_suffix}", y=0.995)
    fig.tight_layout()
    out2 = plot_dir / "vdc_trapping_purity_composition.png"
    fig.savefig(out2)
    plt.close(fig)

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
