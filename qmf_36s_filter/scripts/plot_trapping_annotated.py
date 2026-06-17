"""Final annotated two-panel trapped-fraction vs Vdc figure with selective bands."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]
SCENARIOS = ["monoenergetic", "realistic_spread"]

# selective windows where the target traps and the other two are ~zero (raw)
BAND_36S = (27.75, 28.75)   # clean 36S
BAND_32S = (37.25, 39.25)   # clean 32S


def load(p):
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["scenario"], int(r["mass"]), float(r["qmf_dc"]), float(r["trapped_fraction"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title-suffix", default=" at 2.4 MHz, RF 448 Vp (25 ions/pt)")
    args = ap.parse_args()
    rows = load(args.csv)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.4), dpi=160, sharex=True)
    for ax, scenario in zip(axes, SCENARIOS):
        ax.axvspan(*BAND_36S, color=COLORS[36], alpha=0.10, zorder=0)
        ax.axvspan(*BAND_32S, color=COLORS[32], alpha=0.10, zorder=0)
        for mass in MASSES:
            pts = sorted([r for r in rows if r[0] == scenario and r[1] == mass], key=lambda r: r[2])
            if not pts:
                continue
            ax.plot([p[2] for p in pts], [p[3] for p in pts], marker="o", ms=4, lw=1.8,
                    color=COLORS[mass], label=f"{mass}S", zorder=3)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlim(23.5, 40.5)
        ax.set_ylabel("Trapped fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Mass", loc="upper center", ncol=3, fontsize=8)
        ax.set_title(scenario.replace("_", " "))
        ax.text(np.mean(BAND_36S), 1.0, "36S-selective", ha="center", va="top", fontsize=8,
                color=COLORS[36], fontweight="bold")
        ax.text(np.mean(BAND_32S), 1.0, "32S-selective", ha="center", va="top", fontsize=8,
                color=COLORS[32], fontweight="bold")
        ax.text(33.0, 1.0, "overlap\n(32S-dominated by abundance)", ha="center", va="top",
                fontsize=7.5, color="0.35")
    axes[-1].set_xlabel("QMF DC voltage (V)")
    fig.suptitle(f"QMF isotope trapping vs Vdc{args.title_suffix}", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
