"""Map the QMF transmission/stability region in the (a,q) Mathieu plane.

For a single fixed mass, q = 4eVrf/(m r0^2 Omega^2) and a = 8eVdc/(m r0^2 Omega^2),
so scanning (Vrf, Vdc) IS scanning (a, q). Each grid cell is one SIMION run;
we record transmission / trapped / out-of-bounds fractions. Resumable by (q,a).
Plots a heatmap with the analytic Mathieu first-region boundary overlaid.
"""
from __future__ import annotations
import argparse, csv, sys, subprocess, uuid
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

QMF_DIR = Path(r"C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF")
SIMION = Path(r"C:\Program Files\SIMION-8.1\simion.exe")
IOB = QMF_DIR / "QMF.iob"
EVENT_CSV = QMF_DIR / "data" / "trap_events.csv"
E = 1.602176634e-19
AMU = 1.66053907e-27
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def vrf_vdc(q, a, mass, r0_m, Omega):
    k = mass * AMU * r0_m**2 * Omega**2
    return q * k / (4 * E), a * k / (8 * E)


def write_fly2(mass, n, ke_ev, freq_mhz, src_r_mm, cone_deg, fwhm_ke_ev=0.0):
    p = QMF_DIR / "data" / f"_aq_m{mass}_{uuid.uuid4().hex[:8]}.fly2"
    if fwhm_ke_ev and fwhm_ke_ev > 0:
        ke_block = ("ke = gaussian_distribution {\n"
                    f"      mean = {ke_ev},\n      fwhm = {fwhm_ke_ev}\n    }}")
    else:
        ke_block = f"ke = {ke_ev}"
    p.write_text(f"""particles {{
  coordinates = 0,
  standard_beam {{
    n = {n},
    tob = uniform_distribution {{ min = 0.0, max = {1.0/freq_mhz} }},
    mass = {mass}, charge = 1, cwf = 1, color = 0,
    direction = cone_direction_distribution {{ axis = vector(0,1,0), half_angle = {cone_deg}, fill = true }},
    position = circle_distribution {{ center = vector(19,19,19), normal = vector(0,1,0), radius = {src_r_mm}, fill = true }},
    {ke_block}
  }}
}}
""", encoding="ascii")
    return p


def count_events():
    counts = {"timeout_trapped": 0, "reached_stop_plane": 0, "out_of_bounds": 0}
    if not EVENT_CSV.exists():
        return counts
    seen = set()
    with EVENT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ion = row.get("ion")
            if not ion or ion in seen:
                continue
            seen.add(ion)
            if row.get("event") in counts:
                counts[row["event"]] += 1
    return counts


def run_simion(Vrf, Vdc, mass, a):
    fly = write_fly2(mass, a.num_particles if a.chunk_size <= 0 else a.chunk_size,
                     a.mean_ke_ev, a.rf_freq_mhz, a.source_radius_mm, a.half_angle_deg, a.fwhm_ke_ev)
    cmd = [str(SIMION), "--nogui", "--noprompt", "fly",
           "--trajectory-quality", str(a.trajectory_quality),
           "--particles", str(fly), "--grouped", "1", "--programs", "1", "--retain-trajectories", "0",
           "--adjustable", f"RF_freq_MHz={a.rf_freq_mhz}",
           "--adjustable", f"QMF_RF_amp_Vp={Vrf}",
           "--adjustable", f"QMF_DC_V={Vdc}",
           "--adjustable", f"entrance_Brubaker_RF_amp_Vp={Vrf}",
           "--adjustable", f"exit_Brubaker_RF_amp_Vp={Vrf}",
           "--adjustable", f"max_flight_time_us={a.max_flight_time_us}",
           "--adjustable", f"stop_y_mm={a.stop_y_mm}",
           "--adjustable", "min_x_mm=12", "--adjustable", "max_x_mm=26",
           "--adjustable", "min_z_mm=12", "--adjustable", "max_z_mm=26", "--adjustable", "min_y_mm=10"]
    if a.pressure_pa is not None and a.pressure_pa >= 0:
        cmd += ["--adjustable", f"pressure_pa={a.pressure_pa}"]
    cmd.append(str(IOB))
    try:
        subprocess.run(cmd, cwd=QMF_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    finally:
        try: fly.unlink()
        except OSError: pass


def run_point(Vrf, Vdc, mass, a):
    chunk = a.chunk_size if a.chunk_size > 0 else a.num_particles
    tot = {"timeout_trapped": 0, "reached_stop_plane": 0, "out_of_bounds": 0}
    rem = a.num_particles
    while rem > 0:
        n = min(chunk, rem)
        a._cur_n = n  # not used; chunk size handled in write
        run_simion(Vrf, Vdc, mass, a)
        c = count_events()
        for k in tot: tot[k] += c[k]
        rem -= n
    return tot


def grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 6) for i in range(n)]


def load_done(out):
    if not Path(out).exists():
        return set()
    with open(out, newline="", encoding="utf-8") as f:
        return {(round(float(r["q"]), 4), round(float(r["a"]), 4)) for r in csv.DictReader(f)}


def append(out, row):
    exists = Path(out).exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


def mathieu_boundary_mask(qv, av, n=500):
    def trace(A, Q):
        h = np.pi / n
        r0 = np.ones_like(A); r1 = np.zeros_like(A); s0 = np.zeros_like(A); s1 = np.ones_like(A); t = 0.0
        for _ in range(n):
            def d(t, r0, r1, s0, s1):
                c = A - 2*Q*np.cos(2*t); return s0, s1, -c*r0, -c*r1
            k1 = d(t, r0, r1, s0, s1)
            k2 = d(t+h/2, r0+h/2*k1[0], r1+h/2*k1[1], s0+h/2*k1[2], s1+h/2*k1[3])
            k3 = d(t+h/2, r0+h/2*k2[0], r1+h/2*k2[1], s0+h/2*k2[2], s1+h/2*k2[3])
            k4 = d(t+h, r0+h*k3[0], r1+h*k3[1], s0+h*k3[2], s1+h*k3[3])
            r0 = r0+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); r1 = r1+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            s0 = s0+h/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]); s1 = s1+h/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]); t += h
        return np.abs(r0+s1)
    Q, A = np.meshgrid(qv, av)
    return (trace(A, Q) <= 2) & (trace(-A, -Q) <= 2)


def plot(out, plot_path, mass, metric="transmission_fraction"):
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    if not rows: return
    qs = sorted({round(float(r["q"]), 4) for r in rows})
    as_ = sorted({round(float(r["a"]), 4) for r in rows})
    Z = np.full((len(as_), len(qs)), np.nan)
    qi = {q: i for i, q in enumerate(qs)}; ai = {a: i for i, a in enumerate(as_)}
    for r in rows:
        Z[ai[round(float(r["a"]), 4)], qi[round(float(r["q"]), 4)]] = float(r[metric])
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=190)
    dq = (qs[1]-qs[0]) if len(qs) > 1 else 0.05
    da = (as_[1]-as_[0]) if len(as_) > 1 else 0.02
    extent = [qs[0]-dq/2, qs[-1]+dq/2, as_[0]-da/2, as_[-1]+da/2]
    im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label=metric.replace("_", " "))
    # analytic boundary overlay
    qb = np.linspace(0, 1.0, 400); ab = np.linspace(0, 0.30, 300)
    mask = mathieu_boundary_mask(qb, ab)
    ax.contour(qb, ab, mask.astype(float), levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
    ax.set_xlabel("q   ($\\propto V_{rf}$)"); ax.set_ylabel("a   ($\\propto V_{dc}$)")
    ax.set_title(f"QMF {metric.split('_')[0]} in the (a,q) plane — {mass}S, sim\n(white dashed = analytic Mathieu boundary)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], max(extent[3], 0.27))
    fig.tight_layout(); fig.savefig(plot_path); plt.close(fig)
    print(f"Wrote {plot_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=int, default=36)
    ap.add_argument("--r0-mm", type=float, default=5.1942)
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--q-min", type=float, default=0.10); ap.add_argument("--q-max", type=float, default=0.95); ap.add_argument("--q-step", type=float, default=0.05)
    ap.add_argument("--a-min", type=float, default=0.0); ap.add_argument("--a-max", type=float, default=0.26); ap.add_argument("--a-step", type=float, default=0.02)
    ap.add_argument("--num-particles", type=int, default=20); ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--trajectory-quality", type=int, default=1)
    ap.add_argument("--mean-ke-ev", type=float, default=0.7367136539184808)
    ap.add_argument("--source-radius-mm", type=float, default=0.0); ap.add_argument("--half-angle-deg", type=float, default=0.0)
    ap.add_argument("--fwhm-ke-ev", type=float, default=0.0)
    ap.add_argument("--pressure-pa", type=float, default=0.0)
    ap.add_argument("--max-flight-time-us", type=float, default=100.0); ap.add_argument("--stop-y-mm", type=float, default=170.0)
    ap.add_argument("--output", required=True); ap.add_argument("--plot-dir", required=True)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    Path(a.plot_dir).mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        for met in ("transmission_fraction", "trapped_fraction", "survival_fraction"):
            plot(a.output, str(Path(a.plot_dir)/f"aq_{met}_{a.mass}S.png"), a.mass, met)
        return
    Omega = 2*np.pi*a.rf_freq_mhz*1e6
    done = load_done(a.output)
    qs = grid(a.q_min, a.q_max, a.q_step); as_ = grid(a.a_min, a.a_max, a.a_step)
    total = len(qs)*len(as_); k = 0
    for q in qs:
        for av in as_:
            k += 1
            if (round(q, 4), round(av, 4)) in done:
                continue
            Vrf, Vdc = vrf_vdc(q, av, a.mass, a.r0_mm*1e-3, Omega)
            c = run_point(Vrf, Vdc, a.mass, a)
            N = a.num_particles
            row = {"q": q, "a": av, "vrf": round(Vrf, 3), "vdc": round(Vdc, 3), "mass": a.mass,
                   "transmitted": c["reached_stop_plane"], "trapped": c["timeout_trapped"], "oob": c["out_of_bounds"],
                   "transmission_fraction": c["reached_stop_plane"]/N, "trapped_fraction": c["timeout_trapped"]/N,
                   "survival_fraction": (c["reached_stop_plane"]+c["timeout_trapped"])/N, "num_particles": N}
            append(a.output, row)
            print(f"{k}/{total} q={q:.3f} a={av:.3f} Vrf={Vrf:.0f} Vdc={Vdc:.0f}  T={row['transmission_fraction']:.2f} trap={row['trapped_fraction']:.2f}", flush=True)
    for met in ("transmission_fraction", "trapped_fraction", "survival_fraction"):
        plot(a.output, str(Path(a.plot_dir)/f"aq_{met}_{a.mass}S.png"), a.mass, met)


if __name__ == "__main__":
    main()
