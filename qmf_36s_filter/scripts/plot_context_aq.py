"""Context (Vrf,Vdc) landscape re-cast into Mathieu (a,q) coordinates for one mass,
with the analytic first-stability-region boundary overlaid. RGB: 36S=green, 34S=red, 32S=blue."""
from __future__ import annotations
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

E = 1.602176634e-19; AMU = 1.66053907e-27


def load_rgb(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    vrfs = sorted({round(float(r["vrf"]), 2) for r in rows}); vdcs = sorted({round(float(r["vdc"]), 2) for r in rows})
    vi = {v: i for i, v in enumerate(vrfs)}; di = {v: i for i, v in enumerate(vdcs)}
    rgb = np.zeros((len(vdcs), len(vrfs), 3))
    for r in rows:
        i = di[round(float(r["vdc"]), 2)]; j = vi[round(float(r["vrf"]), 2)]
        rgb[i, j] = [float(r["t34"]), float(r["t36"]), float(r["t32"])]
    return np.clip(rgb, 0, 1), vrfs, vdcs


def trace(A, Q, n=500):
    h = np.pi/n; r0 = np.ones_like(A); r1 = np.zeros_like(A); s0 = np.zeros_like(A); s1 = np.ones_like(A); t = 0.0
    for _ in range(n):
        def d(t, r0, r1, s0, s1):
            c = A - 2*Q*np.cos(2*t); return s0, s1, -c*r0, -c*r1
        k1 = d(t, r0, r1, s0, s1); k2 = d(t+h/2, r0+h/2*k1[0], r1+h/2*k1[1], s0+h/2*k1[2], s1+h/2*k1[3])
        k3 = d(t+h/2, r0+h/2*k2[0], r1+h/2*k2[1], s0+h/2*k2[2], s1+h/2*k2[3]); k4 = d(t+h, r0+h*k3[0], r1+h*k3[1], s0+h*k3[2], s1+h*k3[3])
        r0 = r0+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); r1 = r1+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        s0 = s0+h/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]); s1 = s1+h/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]); t += h
    return np.abs(r0+s1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mass", type=int, default=36); ap.add_argument("--r0-mm", type=float, default=8.76)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4); ap.add_argument("--sp", default="1400,125")
    a = ap.parse_args()
    rgb, vrfs, vdcs = load_rgb(a.context)
    Omega = 2*np.pi*a.rf_freq_mhz*1e6; r0 = a.r0_mm*1e-3
    Cq = 4*E/(a.mass*AMU*r0**2*Omega**2)      # q = Cq * Vrf
    Ca = 8*E/(a.mass*AMU*r0**2*Omega**2)      # a = Ca * Vdc
    dv = vrfs[1]-vrfs[0]; dd = vdcs[1]-vdcs[0]
    ext = [Cq*(vrfs[0]-dv/2), Cq*(vrfs[-1]+dv/2), Ca*(vdcs[0]-dd/2), Ca*(vdcs[-1]+dd/2)]

    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=150)
    ax.imshow(rgb, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    # analytic Mathieu first-region boundary
    qb = np.linspace(0, max(1.0, ext[1]), 400); ab = np.linspace(0, max(0.30, ext[3]), 320)
    Q, A = np.meshgrid(qb, ab); mask = (trace(A, Q) <= 2) & (trace(-A, -Q) <= 2)
    ax.contour(qb, ab, mask.astype(float), levels=[0.5], colors="white", linewidths=1.8, linestyles="--")
    ax.plot(0.706, 0.237, "w*", ms=12); ax.annotate("Mathieu apex", (0.706, 0.237), color="white",
              textcoords="offset points", xytext=(5, 4), fontsize=8)
    # setpoint
    sp = tuple(float(x) for x in a.sp.split(","))
    ax.plot(Cq*sp[0], Ca*sp[1], marker="*", ms=18, mfc="gold", mec="k", mew=1.2, zorder=8)
    ax.annotate(f"setpoint\nV_rf={sp[0]:g} V_dc={sp[1]:g}\n(q={Cq*sp[0]:.2f}, a={Ca*sp[1]:.2f})",
                (Cq*sp[0], Ca*sp[1]), textcoords="offset points", xytext=(-115, -6), fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))

    ax.set_xlim(0, ext[1]); ax.set_ylim(0, ext[3])
    ax.set_xlabel(f"q  (³⁶S;  q = 4eV$_{{rf}}$/m r₀²Ω²,  r₀={a.r0_mm:g} mm)")
    ax.set_ylabel("a  (³⁶S;  a = 8eV$_{dc}$/m r₀²Ω²)")
    ax.set_title("Device transmission landscape in Mathieu (a,q) — ³⁶S axes\n"
                 "green=36S only · white dashed = ideal first stability region")
    ax.legend(handles=[Patch(fc=(0,1,0), label="36S only"), Patch(fc=(1,1,1), ec="0.6", label="all three"),
                       Patch(fc=(0,0,1), label="32S only"), Patch(fc=(1,1,0), label="34S+36S"),
                       Patch(fc=(0,0,0), label="none")], loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
