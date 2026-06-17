"""Survival (transmitted + trapped) vs DC voltage at fixed V_rf, showing the
isotopes peeling off one at a time -> sharp single-isotope selection."""
from __future__ import annotations
import argparse, csv
import matplotlib.pyplot as plt

COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="monoenergetic")
    ap.add_argument("--vrf", type=float, default=404.0)
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] != args.scenario:
                continue
            rows.append((int(r["mass"]), float(r["qmf_dc"]),
                         float(r["transmission_fraction"]) + float(r["trapped_fraction"])))

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    # shade the clean 32S-only window
    ax.axvspan(31.0, 33.0, color=COLORS[32], alpha=0.10, zorder=0)
    ax.text(32.0, 1.05, "32S-only", ha="center", va="bottom", fontsize=8.5,
            color=COLORS[32], fontweight="bold")

    for m in MASSES:
        pts = sorted([(u, s) for mm, u, s in rows if mm == m])
        if not pts:
            continue
        ax.plot([u for u, _ in pts], [s for _, s in pts], marker="o", ms=5, lw=2.0,
                color=COLORS[m], label=f"{m}S")

    ax.set_xlabel("QMF DC voltage U (V)")
    ax.set_ylabel("Survival fraction  (transmitted + trapped)")
    ax.set_ylim(-0.04, 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Mass", loc="lower left")
    ax.set_title(f"Isotope survival vs DC at fixed V_rf = {args.vrf:g} V\n"
                 "(2.4 MHz, ideal: p=0, source r=0, cone=0) — isotopes peel off ~2 V apart")
    # annotate ejection order
    ax.annotate("36S ejected", (30, 0.0), xytext=(30, 0.30), ha="center", fontsize=8,
                color=COLORS[36], arrowprops=dict(arrowstyle="->", color=COLORS[36]))
    ax.annotate("34S ejected", (32, 0.0), xytext=(34.5, 0.55), ha="center", fontsize=8,
                color=COLORS[34], arrowprops=dict(arrowstyle="->", color=COLORS[34]))
    ax.annotate("32S ejected", (34, 0.0), xytext=(37, 0.30), ha="center", fontsize=8,
                color=COLORS[32], arrowprops=dict(arrowstyle="->", color=COLORS[32]))
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
