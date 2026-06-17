"""Single writeup figure: (A) context landscape, (B) detail window + setpoint star,
(C) penetration diagnostic (embedded), (D) summary box with validated numbers."""
from __future__ import annotations
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, ConnectionPatch, Patch


def load_rgb(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    vrfs = sorted({round(float(r["vrf"]), 2) for r in rows}); vdcs = sorted({round(float(r["vdc"]), 2) for r in rows})
    vi = {v: i for i, v in enumerate(vrfs)}; di = {v: i for i, v in enumerate(vdcs)}
    rgb = np.zeros((len(vdcs), len(vrfs), 3))
    for r in rows:
        i = di[round(float(r["vdc"]), 2)]; j = vi[round(float(r["vrf"]), 2)]
        rgb[i, j] = [float(r["t34"]), float(r["t36"]), float(r["t32"])]
    rgb = np.clip(rgb, 0, 1)
    dv = (vrfs[1]-vrfs[0]) if len(vrfs) > 1 else 40; dd = (vdcs[1]-vdcs[0]) if len(vdcs) > 1 else 10
    return rgb, [vrfs[0]-dv/2, vrfs[-1]+dv/2, vdcs[0]-dd/2, vdcs[-1]+dd/2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True); ap.add_argument("--detail", required=True)
    ap.add_argument("--penetration-png", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sp", default="1400,125")
    a = ap.parse_args()
    sp = tuple(float(x) for x in a.sp.split(","))
    cR, cE = load_rgb(a.context); dR, dE = load_rgb(a.detail)

    fig = plt.figure(figsize=(13.5, 10.5), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.28, wspace=0.22)
    axC = fig.add_subplot(gs[0, 0]); axD = fig.add_subplot(gs[0, 1])
    axP = fig.add_subplot(gs[1, 0]); axT = fig.add_subplot(gs[1, 1])

    # A: context
    axC.imshow(cR, origin="lower", extent=cE, aspect="auto", interpolation="nearest")
    axC.add_patch(Rectangle((dE[0], dE[2]), dE[1]-dE[0], dE[3]-dE[2], fill=False, ec="white", lw=2))
    axC.plot(*sp, marker="*", ms=15, mfc="gold", mec="k", mew=1)
    axC.set_title("(A) Full (V$_{rf}$,V$_{dc}$) landscape", fontsize=11)
    axC.legend(handles=[Patch(fc=(0,1,0), label="36S only"), Patch(fc=(1,1,1), ec="0.6", label="all three"),
                        Patch(fc=(0,0,1), label="32S only"), Patch(fc=(1,1,0), label="34S+36S"),
                        Patch(fc=(0,0,0), label="none")], loc="lower right", fontsize=7)

    # B: detail + star
    axD.imshow(dR, origin="lower", extent=dE, aspect="auto", interpolation="nearest")
    axD.plot(*sp, marker="*", ms=20, mfc="gold", mec="k", mew=1.3, zorder=8)
    axD.annotate(f"setpoint {sp[0]:g}/{sp[1]:g} V", sp, textcoords="offset points", xytext=(9, 7),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))
    axD.set_title("(B) 36S-only window (detail, 40 ions/pt)", fontsize=11)
    for ax in (axC, axD):
        ax.set_xlabel("V$_{rf}$ (V)"); ax.set_ylabel("V$_{dc}$ (V)")
    for y in (dE[2], dE[3]):
        fig.add_artist(ConnectionPatch((dE[1], y), (dE[0], y), "data", "data", axesA=axC, axesB=axD,
                                       color="white", lw=0.9, ls=(0, (4, 3))))

    # C: penetration (embedded)
    axP.imshow(mpimg.imread(a.penetration_png)); axP.axis("off")
    axP.set_title("(C) Where each isotope is lost", fontsize=11)

    # D: summary
    axT.axis("off")
    txt = ("36S TRANSMISSION FILTER — validated\n"
           "fixed RF 2.4 MHz; beam r=0.5 mm, div 1.0°, KE 0.737±0.11 eV\n"
           "──────────────────────────────\n"
           "Setpoint:  V_rf = 1400 V,  V_dc = 125 V\n\n"
           "• 36S:  91% transmitted (182/200), reaches 170 mm exit\n"
           "• 32S:  0/200   ejected at ~40 mm (filter entrance)\n"
           "• 34S:  0/200   ejected at ~50 mm (filter entrance)\n\n"
           "Robust window:  V_rf 1380–1430 V,  V_dc 110–150 V\n\n"
           "Abundance-weighted (nat. S 32:34:36 = 95:4.25:0.01%):\n"
           "  off-mass ejected in first ~10 mm → exponential\n"
           "  suppression (≪ the <1.5% counting bound).\n"
           "  → transmitted beam is 36S-dominated.")
    axT.text(0.0, 0.98, txt, va="top", ha="left", fontsize=10.5, family="monospace",
             transform=axT.transAxes, bbox=dict(boxstyle="round", fc="#f4f7fb", ec="0.6"))

    fig.suptitle("36S isotope-selective transmission in the QMF (SIMION)", fontsize=14, y=0.98)
    fig.savefig(a.out, bbox_inches="tight"); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
