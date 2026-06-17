"""Analytic Mathieu (a,q) first-stability-region diagram for the sinusoidal QMF,
with the 32/34/36S operating lines marked.

Quad field: rods at +/-(U - V cos(Omega t)), field radius r0.
  a = 8 e U / (m r0^2 Omega^2),  q = 4 e V / (m r0^2 Omega^2)
x-motion: Mathieu (a, q); y-motion: Mathieu (-a, -q). Stable if BOTH have
|trace(monodromy)| <= 2 over one period.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

E = 1.602176634e-19
AMU = 1.66053907e-27
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}


def monodromy_trace_grid(A, Q, n=600):
    """|trace| of period-pi monodromy for u''+(A-2Q cos2t)u=0, vectorized over arrays A,Q."""
    h = np.pi / n
    # fundamental matrix elements (each same shape as A): [[r0,r1],[s0,s1]], start = identity
    r0 = np.ones_like(A); r1 = np.zeros_like(A)
    s0 = np.zeros_like(A); s1 = np.ones_like(A)

    def deriv(t, r0, r1, s0, s1):
        c = A - 2.0 * Q * np.cos(2.0 * t)
        # Y' = [[s0,s1],[-c*r0,-c*r1]]
        return s0, s1, -c * r0, -c * r1

    t = 0.0
    for _ in range(n):
        a1 = deriv(t, r0, r1, s0, s1)
        a2 = deriv(t + h/2, r0 + h/2*a1[0], r1 + h/2*a1[1], s0 + h/2*a1[2], s1 + h/2*a1[3])
        a3 = deriv(t + h/2, r0 + h/2*a2[0], r1 + h/2*a2[1], s0 + h/2*a2[2], s1 + h/2*a2[3])
        a4 = deriv(t + h,   r0 + h*a3[0],   r1 + h*a3[1],   s0 + h*a3[2],   s1 + h*a3[3])
        r0 = r0 + h/6*(a1[0] + 2*a2[0] + 2*a3[0] + a4[0])
        r1 = r1 + h/6*(a1[1] + 2*a2[1] + 2*a3[1] + a4[1])
        s0 = s0 + h/6*(a1[2] + 2*a2[2] + 2*a3[2] + a4[2])
        s1 = s1 + h/6*(a1[3] + 2*a2[3] + 2*a3[3] + a4[3])
        t += h
    return np.abs(r0 + s1)


def stable_grid(qv, av):
    """boolean grid (len(av) x len(qv)): both x and y stable."""
    Q, A = np.meshgrid(qv, av)
    tx = monodromy_trace_grid(A, Q)
    ty = monodromy_trace_grid(-A, -Q)
    return (tx <= 2.0) & (ty <= 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rf-amp-v", type=float, default=448.0)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--r0-mm", type=float, default=5.1942)
    ap.add_argument("--vdc-windows", default="32:37.5-39,34:35.5-36.5,36:27.75-29.0")
    args = ap.parse_args()

    Omega = 2 * np.pi * args.rf_freq_mhz * 1e6
    r0 = args.r0_mm * 1e-3
    V = args.rf_amp_v

    def q_of(mass):
        return 4 * E * V / (mass * AMU * r0**2 * Omega**2)

    masses = [32, 34, 36]
    qmass = {m: q_of(m) for m in masses}
    print("Mathieu q per mass (V=%.0f, f=%.2f MHz, r0=%.3f mm):" % (V, args.rf_freq_mhz, args.r0_mm))
    for m in masses:
        print(f"  {m}S: q = {qmass[m]:.4f}")

    # stability region
    qv = np.linspace(0.0, 1.0, 240)
    av = np.linspace(-0.30, 0.30, 200)
    mask = stable_grid(qv, av)

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=160)
    ax.contourf(qv, av, mask.astype(float), levels=[0.5, 1.5], colors=["#cfe8ff"], alpha=0.9)
    ax.contour(qv, av, mask.astype(float), levels=[0.5], colors=["#3b7dd8"], linewidths=1.2)

    # parse windows -> a-range per mass via a = 2*(U/V)*q
    win = {}
    for tok in args.vdc_windows.split(","):
        m, rng = tok.split(":")
        lo, hi = (float(x) for x in rng.split("-"))
        win[int(m)] = (lo, hi)

    for m in masses:
        q = qmass[m]
        ax.axvline(q, color=COLORS[m], ls="--", lw=1.3, alpha=0.8)
        # full operating segment over the scanned 24-40 V
        a_lo = 2 * (24.0 / V) * q
        a_hi = 2 * (40.0 / V) * q
        ax.plot([q, q], [a_lo, a_hi], color=COLORS[m], lw=3.0, alpha=0.30, solid_capstyle="round")
        # measured trapping window highlighted
        if m in win:
            lo, hi = win[m]
            ax.plot([q, q], [2 * (lo / V) * q, 2 * (hi / V) * q],
                    color=COLORS[m], lw=5.5, solid_capstyle="round",
                    label=f"{m}S  q={q:.3f}  (trap {lo:g}-{hi:g} V)")
    # apex marker
    ax.plot(0.706, 0.237, "k*", ms=11)
    ax.annotate("apex (0.706, 0.237)", (0.706, 0.237), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("q  (RF)")
    ax.set_ylabel("a  (DC)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.30, 0.30)
    ax.set_title("QMF first stability region — 32/34/36S operating lines\n"
                 f"V={V:g} V, f={args.rf_freq_mhz:g} MHz, r0={args.r0_mm:.3f} mm")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
