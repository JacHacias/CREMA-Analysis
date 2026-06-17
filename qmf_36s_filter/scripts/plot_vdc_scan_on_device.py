"""Vertical-line Vdc-scan (fixed Vrf) overlaid on the MEASURED device transmission
region (from the (a,q) acceptance map) -- the empirical analog of the ideal-triangle
vertical-scan figure. Connects the (a,q) plane to a real Vdc-scan."""
from __future__ import annotations
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt

E = 1.602176634e-19; AMU = 1.66053907e-27
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-csv", required=True)        # device (a,q) transmission map (e.g. 36S)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vrf", type=float, default=448.0)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--r0-mm", type=float, default=5.1942)
    ap.add_argument("--map-mass", type=int, default=36)
    a = ap.parse_args()
    Omega = 2*np.pi*a.rf_freq_mhz*1e6; r0 = a.r0_mm*1e-3
    qof = lambda m: 4*E*a.vrf/(m*AMU*r0**2*Omega**2)
    aof = lambda m, vdc: 8*E*vdc/(m*AMU*r0**2*Omega**2)

    rows = list(csv.DictReader(open(a.map_csv, newline="", encoding="utf-8")))
    qs = sorted({round(float(r["q"]),4) for r in rows}); as_ = sorted({round(float(r["a"]),4) for r in rows})
    Z = np.full((len(as_),len(qs)), np.nan)
    qi={q:i for i,q in enumerate(qs)}; ai={v:i for i,v in enumerate(as_)}
    for r in rows: Z[ai[round(float(r["a"]),4)], qi[round(float(r["q"]),4)]] = float(r["transmission_fraction"])

    fig, ax = plt.subplots(figsize=(8.6,6.4), dpi=190)
    dq=(qs[1]-qs[0]); da=(as_[1]-as_[0])
    extent=[qs[0]-dq/2, qs[-1]+dq/2, as_[0]-da/2, as_[-1]+da/2]
    im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="transmission fraction")

    vdc_hi = 42.0
    for m in MASSES:
        q = qof(m); a_top = aof(m, vdc_hi)
        ls = "-" if m == a.map_mass else "--"
        ax.plot([q,q],[0,a_top], color=COLORS[m], lw=2.6, ls=ls, zorder=6)
        ax.annotate(f"{m}S", (q, min(a_top, extent[3])), xytext=(0,5), textcoords="offset points",
                    ha="center", color=COLORS[m], fontweight="bold", fontsize=11)
    # selective points from the Vdc-scan trapping result (for reference)
    ax.scatter([qof(36)],[aof(36,28)], s=60, color=COLORS[36], edgecolors="w", zorder=8)
    ax.scatter([qof(32)],[aof(32,38)], s=60, color=COLORS[32], edgecolors="w", zorder=8)

    # right-side Vdc axis for the 36S line (a -> Vdc for the map mass)
    ax2 = ax.twinx()
    ax2.set_ylim(0, aof(a.map_mass, 0) if False else ax.get_ylim()[1])
    # convert a-range to Vdc for map_mass: Vdc = a * m r0^2 Omega^2 / (8e)
    k = a.map_mass*AMU*r0**2*Omega**2/(8*E)
    ax.set_ylim(extent[2], extent[3])
    ax2.set_ylim(extent[2]*k, extent[3]*k)
    ax2.set_ylabel(f"V_dc (V) for {a.map_mass}S at V_rf={a.vrf:g}")

    ax.set_xlim(extent[0], extent[1])
    ax.set_xlabel("q  ($\\propto V_{rf}/m$)"); ax.set_ylabel("a  ($\\propto V_{dc}/m$)")
    ax.set_title(f"V_dc-scan (fixed V_rf={a.vrf:g} V) on the MEASURED {a.map_mass}S transmission region\n"
                 f"solid = {a.map_mass}S (self-consistent); dashed = others (assume mass-independence, pending)")
    fig.tight_layout(); fig.savefig(a.out); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
