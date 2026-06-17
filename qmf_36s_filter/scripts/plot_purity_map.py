"""Visualize a (Vrf, Vdc) purity scan so the 36S-only region is obvious.
Left: RGB composite (R=34S, G=36S, B=32S transmission) -> 36S-only = green,
all-transmit = white, nothing = black. Right: clean-36S = T36*(1-T34)*(1-T32)."""
from __future__ import annotations
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_grid(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    vrfs = sorted({round(float(r["vrf"]), 2) for r in rows})
    vdcs = sorted({round(float(r["vdc"]), 2) for r in rows})
    vi = {v: i for i, v in enumerate(vrfs)}; di = {v: i for i, v in enumerate(vdcs)}
    shape = (len(vdcs), len(vrfs))
    T = {m: np.full(shape, np.nan) for m in (32, 34, 36)}
    clean = np.full(shape, np.nan)
    for r in rows:
        i = di[round(float(r["vdc"]), 2)]; j = vi[round(float(r["vrf"]), 2)]
        T[32][i, j] = float(r["t32"]); T[34][i, j] = float(r["t34"]); T[36][i, j] = float(r["t36"])
        clean[i, j] = float(r["clean36"])
    return vrfs, vdcs, T, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    vrfs, vdcs, T, clean = load_grid(a.csv)
    dv = (vrfs[1]-vrfs[0]) if len(vrfs) > 1 else 40; dd = (vdcs[1]-vdcs[0]) if len(vdcs) > 1 else 10
    ext = [vrfs[0]-dv/2, vrfs[-1]+dv/2, vdcs[0]-dd/2, vdcs[-1]+dd/2]

    # RGB composite: R=34S, G=36S, B=32S
    rgb = np.zeros((len(vdcs), len(vrfs), 3))
    rgb[..., 0] = np.nan_to_num(T[34]); rgb[..., 1] = np.nan_to_num(T[36]); rgb[..., 2] = np.nan_to_num(T[32])
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=130)
    axes[0].imshow(rgb, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    axes[0].set_title("Which isotopes transmit (color = transmission)")
    legend = [Patch(fc=(0, 1, 0), label="36S only (target)"),
              Patch(fc=(1, 1, 1), ec="0.6", label="all three"),
              Patch(fc=(0, 0, 1), label="32S only"),
              Patch(fc=(1, 0, 0), label="34S only"),
              Patch(fc=(1, 1, 0), label="34S+36S"), Patch(fc=(0, 0, 0), label="none")]
    axes[0].legend(handles=legend, loc="upper left", fontsize=7.5, framealpha=0.9)

    im = axes[1].imshow(np.nan_to_num(clean), origin="lower", extent=ext, aspect="auto",
                        cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], label="clean-36S = T36·(1−T34)·(1−T32)")
    axes[1].set_title("36S-only quality")
    for ax in axes:
        ax.set_xlabel("$V_{rf}$ (V)"); ax.set_ylabel("$V_{dc}$ (V)")
    if a.title:
        fig.suptitle(a.title, y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight"); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
