"""2-D (a,q) Mathieu picture exhibiting isotope-selective operation:
the three isotopes lie on one ray a/q = 2 Vdc/Vrf; scaling Vrf slides them
along it, so any target can be placed on the apex while neighbours fall outside.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

E = 1.602176634e-19
AMU = 1.66053907e-27
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
MASSES = [32, 34, 36]
APEX_Q, APEX_A = 0.706, 0.237
APEX_SLOPE = APEX_A / APEX_Q  # = 2 Vdc/Vrf at the apex


def monodromy_trace(A, Q, n=600):
    h = np.pi / n
    r0 = np.ones_like(A); r1 = np.zeros_like(A); s0 = np.zeros_like(A); s1 = np.ones_like(A)
    t = 0.0
    for _ in range(n):
        def d(t, r0, r1, s0, s1):
            c = A - 2.0 * Q * np.cos(2.0 * t)
            return s0, s1, -c * r0, -c * r1
        a1 = d(t, r0, r1, s0, s1)
        a2 = d(t + h/2, r0+h/2*a1[0], r1+h/2*a1[1], s0+h/2*a1[2], s1+h/2*a1[3])
        a3 = d(t + h/2, r0+h/2*a2[0], r1+h/2*a2[1], s0+h/2*a2[2], s1+h/2*a2[3])
        a4 = d(t + h,   r0+h*a3[0],   r1+h*a3[1],   s0+h*a3[2],   s1+h*a3[3])
        r0 = r0 + h/6*(a1[0]+2*a2[0]+2*a3[0]+a4[0]); r1 = r1 + h/6*(a1[1]+2*a2[1]+2*a3[1]+a4[1])
        s0 = s0 + h/6*(a1[2]+2*a2[2]+2*a3[2]+a4[2]); s1 = s1 + h/6*(a1[3]+2*a2[3]+2*a3[3]+a4[3])
        t += h
    return np.abs(r0 + s1)


def is_stable(q, a):
    A = np.array([a]); Q = np.array([q])
    return (monodromy_trace(A, Q)[0] <= 2.0) and (monodromy_trace(-A, -Q)[0] <= 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=34, help="isotope placed on the apex")
    args = ap.parse_args()

    # stability region (high resolution -> smooth boundary)
    qv = np.linspace(0.0, 1.0, 700); av = np.linspace(0.0, 0.30, 480)
    Q, A = np.meshgrid(qv, av)
    mask = (monodromy_trace(A, Q) <= 2.0) & (monodromy_trace(-A, -Q) <= 2.0)

    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=200)
    ax.contourf(qv, av, mask.astype(float), levels=[0.5, 1.5], colors=["#dcebfb"]) \
        if False else ax.contourf(qv, av, mask.astype(float), levels=[0.5, 1.5], colors=["#e8f1fb"])
    ax.contour(qv, av, mask.astype(float), levels=[0.5], colors=["#2f6fd0"], linewidths=1.8)
    ax.plot(APEX_Q, APEX_A, "k*", ms=12, zorder=6)

    # operating ray a = APEX_SLOPE * q
    qline = np.linspace(0, 0.97, 200)
    ax.plot(qline, APEX_SLOPE * qline, color="0.35", ls=(0, (6, 4)), lw=1.4, zorder=4)

    # triplet with TARGET on the apex
    t = args.target
    for m in MASSES:
        q = APEX_Q * t / m
        a = APEX_SLOPE * q
        selected = (m == t)
        ax.scatter([q], [a], s=(170 if selected else 90), color=COLORS[m],
                   edgecolors=("k" if selected else "white"), linewidths=1.4, zorder=7)
        dy = 12 if selected else (-15 if m > t else 13)
        ax.annotate(f"{m}S", (q, a), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=11, color=COLORS[m], fontweight="bold")

    # minimal scan arrow along the ray
    ax.annotate("", xy=(0.88, APEX_SLOPE*0.88), xytext=(0.60, APEX_SLOPE*0.60),
                arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1.6))
    ax.text(0.86, APEX_SLOPE*0.86 - 0.012, "scan $V_{rf}$", color="0.45", fontsize=9,
            ha="right", va="top", rotation=18)

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.275)
    ax.set_xlabel("q   ($\\propto V_{rf}/m$)", fontsize=11)
    ax.set_ylabel("a   ($\\propto V_{dc}/m$)", fontsize=11)
    ax.set_title(f"Selecting {t}S at the apex   (ray: a/q = 2$V_{{dc}}/V_{{rf}}$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
