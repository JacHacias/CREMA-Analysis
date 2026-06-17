"""Hunt for a (Vrf, Vdc) region that transmits ONLY 36S.
At each (Vrf, Vdc) grid point, fly 32S/34S/36S and record transmission.
Score = abundance-weighted 36S purity of the transmitted beam.
Resumable by (Vrf, Vdc). Plots per-mass transmission + purity heatmaps."""
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
ABUND = {32: 0.9499, 34: 0.0425, 36: 0.0001}   # natural S (33S not modelled)
MASSES = [32, 34, 36]
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def write_fly2(mass, n, ke_ev, freq_mhz, src_r_mm, cone_deg, fwhm_ke_ev):
    p = QMF_DIR / "data" / f"_pur_m{mass}_{uuid.uuid4().hex[:8]}.fly2"
    ke = (f"ke = gaussian_distribution {{\n      mean = {ke_ev},\n      fwhm = {fwhm_ke_ev}\n    }}"
          if fwhm_ke_ev and fwhm_ke_ev > 0 else f"ke = {ke_ev}")
    p.write_text(f"""particles {{
  coordinates = 0,
  standard_beam {{
    n = {n},
    tob = uniform_distribution {{ min = 0.0, max = {1.0/freq_mhz} }},
    mass = {mass}, charge = 1, cwf = 1, color = 0,
    direction = cone_direction_distribution {{ axis = vector(0,1,0), half_angle = {cone_deg}, fill = true }},
    position = circle_distribution {{ center = vector(19,19,19), normal = vector(0,1,0), radius = {src_r_mm}, fill = true }},
    {ke}
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
        for r in csv.DictReader(f):
            ion = r.get("ion")
            if not ion or ion in seen:
                continue
            seen.add(ion)
            if r.get("event") in counts:
                counts[r["event"]] += 1
    return counts


def run_mass(Vrf, Vdc, mass, a):
    chunk = a.chunk_size if a.chunk_size > 0 else a.num_particles
    tot = {"timeout_trapped": 0, "reached_stop_plane": 0, "out_of_bounds": 0}
    rem = a.num_particles
    while rem > 0:
        n = min(chunk, rem)
        fly = write_fly2(mass, n, a.mean_ke_ev, a.rf_freq_mhz, a.source_radius_mm, a.half_angle_deg, a.fwhm_ke_ev)
        cmd = [str(SIMION), "--nogui", "--noprompt", "fly", "--trajectory-quality", str(a.trajectory_quality),
               "--particles", str(fly), "--grouped", "1", "--programs", "1", "--retain-trajectories", "0",
               "--adjustable", f"RF_freq_MHz={a.rf_freq_mhz}", "--adjustable", f"QMF_RF_amp_Vp={Vrf}",
               "--adjustable", f"QMF_DC_V={Vdc}", "--adjustable", f"entrance_Brubaker_RF_amp_Vp={Vrf}",
               "--adjustable", f"exit_Brubaker_RF_amp_Vp={Vrf}", "--adjustable", f"max_flight_time_us={a.max_flight_time_us}",
               "--adjustable", f"stop_y_mm={a.stop_y_mm}", "--adjustable", "min_x_mm=12", "--adjustable", "max_x_mm=26",
               "--adjustable", "min_z_mm=12", "--adjustable", "max_z_mm=26", "--adjustable", "min_y_mm=10",
               "--adjustable", f"pressure_pa={a.pressure_pa}", str(IOB)]
        try:
            subprocess.run(cmd, cwd=QMF_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
            c = count_events()
            for k in tot: tot[k] += c[k]
        finally:
            try: fly.unlink()
            except OSError: pass
        rem -= n
    return tot


def grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def load_done(out):
    if not Path(out).exists():
        return set()
    with open(out, newline="", encoding="utf-8") as f:
        return {(round(float(r["vrf"]), 2), round(float(r["vdc"]), 2)) for r in csv.DictReader(f)}


def append(out, row):
    exists = Path(out).exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


def plot(out, plot_dir, metric, label):
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    if not rows: return
    vrfs = sorted({round(float(r["vrf"]), 2) for r in rows})
    vdcs = sorted({round(float(r["vdc"]), 2) for r in rows})
    Z = np.full((len(vdcs), len(vrfs)), np.nan)
    vi = {v: i for i, v in enumerate(vrfs)}; di = {v: i for i, v in enumerate(vdcs)}
    for r in rows:
        Z[di[round(float(r["vdc"]), 2)], vi[round(float(r["vrf"]), 2)]] = float(r[metric])
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=190)
    dv = (vrfs[1]-vrfs[0]) if len(vrfs) > 1 else 40; dd = (vdcs[1]-vdcs[0]) if len(vdcs) > 1 else 5
    ext = [vrfs[0]-dv/2, vrfs[-1]+dv/2, vdcs[0]-dd/2, vdcs[-1]+dd/2]
    im = ax.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("$V_{rf}$ (V)"); ax.set_ylabel("$V_{dc}$ (V)")
    ax.set_title(f"{label} vs ($V_{{rf}}$, $V_{{dc}}$) — sim")
    fig.tight_layout(); fig.savefig(Path(plot_dir)/f"purity_{metric}.png"); plt.close(fig)
    print(f"Wrote purity_{metric}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-freq-mhz", type=float, default=2.4)
    ap.add_argument("--vrf-min", type=float, default=480); ap.add_argument("--vrf-max", type=float, default=720); ap.add_argument("--vrf-step", type=float, default=60)
    ap.add_argument("--vdc-min", type=float, default=15); ap.add_argument("--vdc-max", type=float, default=65); ap.add_argument("--vdc-step", type=float, default=10)
    ap.add_argument("--num-particles", type=int, default=15); ap.add_argument("--chunk-size", type=int, default=15)
    ap.add_argument("--trajectory-quality", type=int, default=1)
    ap.add_argument("--mean-ke-ev", type=float, default=0.7367136539184808); ap.add_argument("--fwhm-ke-ev", type=float, default=0.11451862433053576)
    ap.add_argument("--source-radius-mm", type=float, default=0.5); ap.add_argument("--half-angle-deg", type=float, default=1.0)
    ap.add_argument("--pressure-pa", type=float, default=0.0)
    ap.add_argument("--max-flight-time-us", type=float, default=200.0); ap.add_argument("--stop-y-mm", type=float, default=170.0)
    ap.add_argument("--output", required=True); ap.add_argument("--plot-dir", required=True)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    Path(a.plot_dir).mkdir(parents=True, exist_ok=True)
    mets = [("t36", "36S transmission"), ("t34", "34S transmission"), ("t32", "32S transmission"),
            ("purity36", "36S purity (abundance-wtd)"), ("clean36", "clean-36S = T36*(1-T34)*(1-T32)")]
    if a.plot_only:
        for m, l in mets: plot(a.output, a.plot_dir, m, l)
        return
    done = load_done(a.output)
    vrfs = grid(a.vrf_min, a.vrf_max, a.vrf_step); vdcs = grid(a.vdc_min, a.vdc_max, a.vdc_step)
    total = len(vrfs)*len(vdcs); k = 0
    N = a.num_particles
    for vrf in vrfs:
        for vdc in vdcs:
            k += 1
            if (round(vrf, 2), round(vdc, 2)) in done: continue
            T = {}; tr = {}
            for m in MASSES:
                c = run_mass(vrf, vdc, m, a)
                T[m] = c["reached_stop_plane"]/N; tr[m] = c["timeout_trapped"]/N
            denom = sum(T[m]*ABUND[m] for m in MASSES)
            purity36 = (T[36]*ABUND[36]/denom) if denom > 0 else 0.0
            clean36 = T[36]*(1-T[34])*(1-T[32])
            row = {"vrf": vrf, "vdc": vdc, "t32": T[32], "t34": T[34], "t36": T[36],
                   "trap32": tr[32], "trap34": tr[34], "trap36": tr[36],
                   "purity36": purity36, "clean36": clean36, "num_particles": N}
            append(a.output, row)
            print(f"{k}/{total} Vrf={vrf:.0f} Vdc={vdc:.0f}  T36={T[36]:.2f} T34={T[34]:.2f} T32={T[32]:.2f}  clean36={clean36:.2f} purity36={purity36:.3f}", flush=True)
    for m, l in mets: plot(a.output, a.plot_dir, m, l)


if __name__ == "__main__":
    main()
