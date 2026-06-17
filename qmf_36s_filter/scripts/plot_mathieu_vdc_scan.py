"""Correct (a,q) illustration of a FIXED-Vrf, Vdc-scan (what the simulation does):
each isotope has fixed q (q propto Vrf/m); scanning Vdc moves it straight UP.
Contrast with the apex RAY (Vrf mass-scan) shown faint."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

E = 1.602176634e-19; AMU = 1.66053907e-27
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]
APEX_SLOPE = 0.237/0.706


def trace(A, Q, n=600):
    h = np.pi/n; r0=np.ones_like(A); r1=np.zeros_like(A); s0=np.zeros_like(A); s1=np.ones_like(A); t=0.0
    for _ in range(n):
        def d(t,r0,r1,s0,s1):
            c=A-2*Q*np.cos(2*t); return s0,s1,-c*r0,-c*r1
        k1=d(t,r0,r1,s0,s1); k2=d(t+h/2,r0+h/2*k1[0],r1+h/2*k1[1],s0+h/2*k1[2],s1+h/2*k1[3])
        k3=d(t+h/2,r0+h/2*k2[0],r1+h/2*k2[1],s0+h/2*k2[2],s1+h/2*k2[3]); k4=d(t+h,r0+h*k3[0],r1+h*k3[1],s0+h*k3[2],s1+h*k3[3])
        r0=r0+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); r1=r1+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        s0=s0+h/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]); s1=s1+h/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]); t+=h
    return np.abs(r0+s1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vrf", type=float, default=448.0)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--r0-mm", type=float, default=5.1942)
    a = ap.parse_args()
    Omega = 2*np.pi*a.rf_freq_mhz*1e6; r0 = a.r0_mm*1e-3
    qof = lambda m: 4*E*a.vrf/(m*AMU*r0**2*Omega**2)
    aof = lambda m, vdc: 8*E*vdc/(m*AMU*r0**2*Omega**2)

    qv = np.linspace(0,1.0,260); av = np.linspace(0,0.30,200)
    Q, A = np.meshgrid(qv, av)
    mask = (trace(A,Q)<=2)&(trace(-A,-Q)<=2)

    fig, ax = plt.subplots(figsize=(8.4,6.4), dpi=190)
    ax.contourf(qv, av, mask.astype(float), levels=[0.5,1.5], colors=["#e8f1fb"])
    ax.contour(qv, av, mask.astype(float), levels=[0.5], colors=["#2f6fd0"], linewidths=1.6)
    ax.plot(0.706,0.237,"k*",ms=11)

    # faint apex ray (the Plot-1 mode: scan Vrf)
    ql = np.linspace(0,0.97,50)
    ax.plot(ql, APEX_SLOPE*ql, color="0.6", ls=(0,(5,4)), lw=1.2)
    ax.annotate("Plot-1 mode: scan $V_{rf}$\n(ray, ratio fixed)", (0.93, APEX_SLOPE*0.93),
                xytext=(-4,10), textcoords="offset points", ha="right", fontsize=8.5, color="0.45")

    # THE ACTUAL SIM: fixed Vrf -> vertical lines at fixed q; scan Vdc moves up
    vdc_hi = 42.0
    for m in MASSES:
        q = qof(m); a_top = aof(m, vdc_hi)
        ax.plot([q,q],[0,a_top], color=COLORS[m], lw=3.2, solid_capstyle="round", zorder=5,
                label=f"{m}S  (q={q:.2f} at $V_{{rf}}$={a.vrf:g})")
        ax.annotate(f"{m}S", (q, a_top), xytext=(0,6), textcoords="offset points",
                    ha="center", color=COLORS[m], fontweight="bold", fontsize=10)
    # scan-Vdc arrow on the 34S line
    q34 = qof(34)
    ax.annotate("", xy=(q34, 0.16), xytext=(q34, 0.03), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=2.2))
    ax.annotate("scan $V_{dc}$\n(Plot-2 mode)", (q34, 0.16), xytext=(8,-6), textcoords="offset points",
                fontsize=9, color="0.2")
    # mark the selective points from Plot 2
    ax.scatter([qof(36)],[aof(36,28)], s=70, color=COLORS[36], edgecolors="k", zorder=7)
    ax.annotate("36S selective\n(V$_{dc}$≈28 V)", (qof(36), aof(36,28)), xytext=(-92,-2),
                textcoords="offset points", fontsize=8, color=COLORS[36])
    ax.scatter([qof(32)],[aof(32,38)], s=70, color=COLORS[32], edgecolors="k", zorder=7)
    ax.annotate("32S selective\n(V$_{dc}$≈38 V)", (qof(32), aof(32,38)), xytext=(8,-4),
                textcoords="offset points", fontsize=8, color=COLORS[32])

    ax.set_xlim(0,1.0); ax.set_ylim(0,0.27)
    ax.set_xlabel("q  ($\\propto V_{rf}/m$)"); ax.set_ylabel("a  ($\\propto V_{dc}/m$)")
    ax.set_title(f"What the sim actually scans: fixed $V_{{rf}}$={a.vrf:g} V, sweep $V_{{dc}}$\n"
                 "→ three vertical lines (fixed q), not a ray")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.tight_layout(); fig.savefig(a.out); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
