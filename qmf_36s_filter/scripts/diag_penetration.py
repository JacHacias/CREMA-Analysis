"""Penetration diagnostic at the 36S setpoint: how far does each isotope get
axially before it's lost? Off-mass ions ejected near the entrance = deeply
unstable = strong suppression (a better argument than raw transmission counts).

Runs ONE SIMION fly per isotope (so QMF.lua's y-plane log isn't overwritten),
reads the per-ion max y reached, and plots the penetration distribution."""
from __future__ import annotations
import argparse, csv, subprocess, sys, uuid
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

QMF_DIR = Path(r"C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF")
SIMION = Path(r"C:\Program Files\SIMION-8.1\simion.exe")
IOB = QMF_DIR / "QMF.iob"
DATA_CSV = QMF_DIR / "data" / "data_test.csv"
DATA_FALLBACK = QMF_DIR.parent.parent / "qmf_data_test.csv"
E = 1.602176634e-19; AMU = 1.66053907e-27
COLORS = {32: "#1f77b4", 34: "#d62728", 36: "#2ca02c"}
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def fly2(mass, n, ke, freq, r, cone, fwhm):
    p = QMF_DIR / "data" / f"_pen_m{mass}_{uuid.uuid4().hex[:8]}.fly2"
    keb = (f"ke = gaussian_distribution {{ mean = {ke}, fwhm = {fwhm} }}" if fwhm > 0 else f"ke = {ke}")
    p.write_text(f"""particles {{
  coordinates = 0,
  standard_beam {{
    n = {n}, tob = uniform_distribution {{ min = 0.0, max = {1.0/freq} }},
    mass = {mass}, charge = 1, cwf = 1, color = 0,
    direction = cone_direction_distribution {{ axis = vector(0,1,0), half_angle = {cone}, fill = true }},
    position = circle_distribution {{ center = vector(19,19,19), normal = vector(0,1,0), radius = {r}, fill = true }},
    {keb}
  }}
}}
""", encoding="ascii")
    return p


def run(mass, a):
    f = fly2(mass, a.num_particles, a.mean_ke_ev, a.rf_freq_mhz, a.source_radius_mm, a.half_angle_deg, a.fwhm_ke_ev)
    cmd = [str(SIMION), "--nogui", "--noprompt", "fly", "--trajectory-quality", "1",
           "--particles", str(f), "--grouped", "1", "--programs", "1", "--retain-trajectories", "0",
           "--adjustable", f"RF_freq_MHz={a.rf_freq_mhz}", "--adjustable", f"QMF_RF_amp_Vp={a.vrf}",
           "--adjustable", f"QMF_DC_V={a.vdc}", "--adjustable", f"entrance_Brubaker_RF_amp_Vp={a.vrf}",
           "--adjustable", f"exit_Brubaker_RF_amp_Vp={a.vrf}", "--adjustable", f"max_flight_time_us={a.max_flight_time_us}",
           "--adjustable", "stop_y_mm=170", "--adjustable", "min_x_mm=12", "--adjustable", "max_x_mm=26",
           "--adjustable", "min_z_mm=12", "--adjustable", "max_z_mm=26", "--adjustable", "min_y_mm=10",
           "--adjustable", f"pressure_pa={a.pressure_pa}", str(IOB)]
    subprocess.run(cmd, cwd=QMF_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    try: f.unlink()
    except OSError: pass
    path = DATA_CSV if DATA_CSV.exists() else (DATA_FALLBACK if DATA_FALLBACK.exists() else None)
    maxy = {}
    if path:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ion = row.get("ion"); y = row.get("y_plane")
                if ion is None or y is None: continue
                yv = float(y); maxy[ion] = max(maxy.get(ion, 0.0), yv)
    return list(maxy.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrf", type=float, default=1400); ap.add_argument("--vdc", type=float, default=125)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--num-particles", type=int, default=80)
    ap.add_argument("--mean-ke-ev", type=float, default=0.7367136539184808); ap.add_argument("--fwhm-ke-ev", type=float, default=0.11451862433053576)
    ap.add_argument("--source-radius-mm", type=float, default=0.5); ap.add_argument("--half-angle-deg", type=float, default=1.0)
    ap.add_argument("--pressure-pa", type=float, default=0.0); ap.add_argument("--max-flight-time-us", type=float, default=150.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pen = {m: run(m, a) for m in (32, 34, 36)}

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
    for k, m in enumerate((36, 34, 32)):
        ys = np.array(pen[m]); x = np.full(len(ys), k) + (np.random.default_rng(m).uniform(-0.13, 0.13, len(ys)) if len(ys) else 0)
        ax.scatter(x, ys, s=14, color=COLORS[m], alpha=0.6, edgecolors="none")
        if len(ys):
            ax.hlines(np.median(ys), k-0.25, k+0.25, color="k", lw=2)
            ax.text(k, 178, f"{m}S\nmed={np.median(ys):.0f} mm\nn={len(ys)}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(170, color="0.4", ls="--", lw=1); ax.text(2.4, 171, "exit / stop plane (170 mm)", fontsize=8, color="0.4")
    ax.axhspan(41.5, 161.5, color="0.9", zorder=0); ax.text(2.45, 100, "filter section", fontsize=8, color="0.5", rotation=90, va="center")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["36S", "34S", "32S"])
    ax.set_ylabel("max axial penetration before loss (mm)")
    ax.set_ylim(0, 195)
    ax.set_title(f"Where each isotope is lost @ setpoint V_rf={a.vrf:g}, V_dc={a.vdc:g} V\n"
                 "(36S reaches the exit; off-mass ejected early = deeply unstable)")
    fig.tight_layout(); fig.savefig(a.out)
    print(f"Wrote {a.out}")
    for m in (36, 34, 32):
        ys = pen[m]
        print(f"{m}S: n={len(ys)} median_penetration={np.median(ys) if ys else 0:.0f} mm  reached_exit={sum(1 for y in ys if y>=165)}/{len(ys)}")


if __name__ == "__main__":
    main()
