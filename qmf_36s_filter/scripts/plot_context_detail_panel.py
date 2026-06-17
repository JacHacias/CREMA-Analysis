"""Context + detail zoom panel: wide (Vrf,Vdc) landscape on the left with the
detail region boxed, and the high-stats detail map on the right, connected by
zoom lines. RGB composite: 36S->green, 34S->red, 32S->blue."""
from __future__ import annotations
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Patch


def load_rgb(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    vrfs = sorted({round(float(r["vrf"]), 2) for r in rows})
    vdcs = sorted({round(float(r["vdc"]), 2) for r in rows})
    vi = {v: i for i, v in enumerate(vrfs)}; di = {v: i for i, v in enumerate(vdcs)}
    rgb = np.zeros((len(vdcs), len(vrfs), 3))
    for r in rows:
        i = di[round(float(r["vdc"]), 2)]; j = vi[round(float(r["vrf"]), 2)]
        rgb[i, j, 0] = float(r["t34"]); rgb[i, j, 1] = float(r["t36"]); rgb[i, j, 2] = float(r["t32"])
    rgb = np.clip(rgb, 0, 1)
    dv = (vrfs[1]-vrfs[0]) if len(vrfs) > 1 else 40; dd = (vdcs[1]-vdcs[0]) if len(vdcs) > 1 else 10
    ext = [vrfs[0]-dv/2, vrfs[-1]+dv/2, vdcs[0]-dd/2, vdcs[-1]+dd/2]
    return rgb, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True); ap.add_argument("--detail", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--setpoint", default="")  # "Vrf,Vdc" to mark with a star
    a = ap.parse_args()
    sp = None
    if a.setpoint:
        sp = tuple(float(x) for x in a.setpoint.split(","))
    cR, cE = load_rgb(a.context)
    dR, dE = load_rgb(a.detail)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=130,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    axL.imshow(cR, origin="lower", extent=cE, aspect="auto", interpolation="nearest")
    axL.set_title("Context: full (V$_{rf}$, V$_{dc}$) landscape")
    # box around the detail region (in context coords)
    box = Rectangle((dE[0], dE[2]), dE[1]-dE[0], dE[3]-dE[2], fill=False, ec="white", lw=2.0, zorder=5)
    axL.add_patch(box)
    axL.legend(handles=[Patch(fc=(0,1,0), label="36S only (target)"),
                        Patch(fc=(1,1,1), ec="0.6", label="all three"),
                        Patch(fc=(0,0,1), label="32S only"),
                        Patch(fc=(1,1,0), label="34S+36S"),
                        Patch(fc=(0,0,0), label="none")],
               loc="lower right", fontsize=7.5, framealpha=0.92)

    axR.imshow(dR, origin="lower", extent=dE, aspect="auto", interpolation="nearest")
    axR.set_title("Detail: 36S-only window (40 ions/pt)")

    # zoom-fan connection lines: box right edge -> detail panel left edge
    for y in (dE[2], dE[3]):
        con = ConnectionPatch(xyA=(dE[1], y), coordsA=axL.transData,
                              xyB=(dE[0], y), coordsB=axR.transData,
                              color="white", lw=1.0, ls=(0, (4, 3)), zorder=6)
        fig.add_artist(con)

    if sp:
        for ax in (axL, axR):
            ax.plot(sp[0], sp[1], marker="*", ms=18, mfc="gold", mec="k", mew=1.2, zorder=8)
        axR.annotate(f"setpoint\n{sp[0]:g} V / {sp[1]:g} V", sp, textcoords="offset points",
                     xytext=(10, 8), fontsize=9, color="k", fontweight="bold",
                     bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))

    for ax in (axL, axR):
        ax.set_xlabel("V$_{rf}$ (V)"); ax.set_ylabel("V$_{dc}$ (V)")
    fig.suptitle("36S transmission filter @ 2.4 MHz (fixed) — green = 36S only, others rejected", y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight"); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
