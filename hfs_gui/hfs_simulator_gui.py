"""
Local browser GUI for simulating hyperfine (and even-even) resonance-ionization
spectra, built on proposal_simulation.py.

Inputs: isotope label, nuclear spin I, lower/upper electronic J, hyperfine
constants A/B, linewidth (FWHM), stopped-beam rate, efficiency, background, and
measurement time. The app simulates a Poisson-sampled spectrum, fits it the same
way as the real data, and reports the extracted centroid (isotope shift) and,
for odd-A isotopes, the hyperfine A/B constants with their statistical
uncertainties.

Run with:
    .venv\\Scripts\\python.exe hfs_simulator_gui.py
or use launch_hfs_simulator_gui.bat
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Non-interactive backend before pyplot is imported (threaded server workers).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared analysis modules live at the repo root; make them importable no matter the
# working directory (this GUI now lives under hfs_gui/).
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import proposal_simulation as ps
from isotope_shift_analysis import voigt


# --------------------------------------------------------------------------
# Simulation + figure
# --------------------------------------------------------------------------
def run_simulation(params: dict) -> dict:
    """Build a spec from the GUI payload, simulate, fit, and render a PNG."""
    label = str(params.get("isotope", "Isotope")).strip() or "Isotope"
    I = float(params["spin"])
    Jl = float(params["J_lower"])
    Ju = float(params["J_upper"])

    spec = {"I": I, "J_lower": Jl, "J_upper": Ju}
    if I > 0:
        spec.update(
            A_lower=float(params.get("A_lower", 0.0)),
            B_lower=float(params.get("B_lower", 0.0)),
            A_upper=float(params.get("A_upper", 0.0)),
            B_upper=float(params.get("B_upper", 0.0)),
        )

    fwhm = float(params["linewidth"])
    lfrac = float(params.get("lorentz_fraction", 0.3))
    rate = float(params["rate"])
    eff = float(params.get("efficiency", 1e-3))
    bkg = float(params.get("background", 2e-3))
    time_h = float(params.get("time_h", 48.0))
    seed = int(float(params.get("seed", 1)))

    scan = params.get("scan_range")
    if scan in (None, "", "auto"):
        scan_range = ps.auto_scan_range(spec, fwhm)
    else:
        half = abs(float(scan))
        scan_range = (-half, half)

    n_steps = params.get("n_steps")
    n_steps = ps.auto_n_steps(scan_range, fwhm) if n_steps in (None, "", "auto") \
        else int(float(n_steps))

    sim = ps.simulate_spectrum(
        spec, stopped_beam_rate_pps=rate, total_efficiency=eff,
        background_cps=bkg, measurement_time_s=time_h * ps.HOUR,
        scan_range_MHz=scan_range, n_steps=n_steps,
        fwhm_MHz=fwhm, lorentz_fraction=lfrac, seed=seed)

    # Fit (same model as the data) and collect structured results.
    results = {
        "label": label, "even_even": I == 0,
        "peak_counts": round(sim["peak_counts"], 1),
        "total_counts": round(sim["total_counts"], 1),
        "scan_range_MHz": [round(scan_range[0], 1), round(scan_range[1], 1)],
        "n_steps": int(n_steps), "n_components": len(sim["components"]),
        "fit": {}, "fit_ok": False, "fit_message": "",
    }
    xf = np.linspace(sim["x"].min(), sim["x"].max(), 2000)
    yf = None
    try:
        if I == 0:
            center, center_err, popt = ps.fit_even_even(sim["x"], sim["counts"])
            yf = voigt(xf, *popt)
            results["fit"] = {"centroid (MHz)": _pm(center, center_err)}
        else:
            fp, model, popt = ps.fit_hfs(sim)
            yf = model(xf, *popt)
            results["fit"] = {
                "centroid (MHz)": _pm(*fp["centroid"]),
                "A_lower (MHz)": _pm(*fp["A_lower"]),
                "A_upper (MHz)": _pm(*fp["A_upper"]),
                "B_lower (MHz)": _pm(*fp["B_lower"]),
                "B_upper (MHz)": _pm(*fp["B_upper"]),
            }
        results["fit_ok"] = True
    except Exception as exc:  # noqa: BLE001
        results["fit_message"] = f"Fit did not converge: {exc}"

    results["image"] = _render_png(sim, xf, yf, label, spec)
    results["x"] = [round(float(v), 3) for v in sim["x"]]
    results["counts"] = [int(v) for v in sim["counts"]]
    return results


# --------------------------------------------------------------------------
# Presets (from proposal_simulation.ISOTOPES) + calibrated offline linewidth
# --------------------------------------------------------------------------
DEFAULT_CAL_SCAN = (
    "exports/sulfur_library_scan_bundle_20260602_162624/scans/32S_3-23-26.csv"
)
_CAL_CACHE: dict = {"done": False, "value": None}


def build_presets() -> dict:
    presets = {}
    for name, spec in ps.ISOTOPES.items():
        meta = ps.ISOTOPE_META.get(name, {})
        presets[name] = {
            "isotope": name,
            "spin": spec["I"],
            "J_lower": spec["J_lower"],
            "J_upper": spec["J_upper"],
            "A_lower": spec.get("A_lower", 0.0),
            "A_upper": spec.get("A_upper", 0.0),
            "B_lower": spec.get("B_lower", 0.0),
            "B_upper": spec.get("B_upper", 0.0),
            "Jpi": meta.get("Jpi", ""),
        }
    return presets


def get_calibrated_linewidth():
    """FWHM (MHz) from the offline 32S commissioning scan, cached; None if absent."""
    if not _CAL_CACHE["done"]:
        _CAL_CACHE["done"] = True
        try:
            import os
            if os.path.exists(DEFAULT_CAL_SCAN):
                cal = ps.calibrate_from_scan(
                    DEFAULT_CAL_SCAN, isotope="32S", bin_width_MHz=25.0)
                _CAL_CACHE["value"] = round(float(cal["fwhm_MHz"]), 1)
        except Exception:  # noqa: BLE001
            _CAL_CACHE["value"] = None
    return _CAL_CACHE["value"]


def _pm(value: float, err: float) -> str:
    if not np.isfinite(err):
        return f"{value:.2f} ± (n/a)"
    return f"{value:.2f} ± {err:.2f}"


def _render_png(sim, xf, yf, label, spec) -> str:
    ps.apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    counts = sim["counts"]
    yerr = np.sqrt(np.clip(counts, 0.0, None))
    yerr[yerr == 0] = 1.0
    if yf is not None:
        ax.plot(xf, yf, color="C1", lw=2, label="Fit", zorder=3)
    ax.errorbar(sim["x"], counts, yerr=yerr, fmt="o", ms=4.5, color="C0",
                ecolor="black", elinewidth=0.9, capsize=2.0, capthick=0.9,
                label="Data", zorder=2)
    if spec["I"] == 0:
        title = rf"$^{{{_mass(label)}}}${_elem(label)}  (0$^+$)"
    else:
        title = rf"$^{{{_mass(label)}}}${_elem(label)}  (I={_spin(spec['I'])})"
    ax.text(0.04, 0.92, title, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top")
    ax.set_xlabel("Relative frequency (MHz)", fontweight="bold")
    ax.set_ylabel("Counts", fontweight="bold")
    ps.style_axes(ax)
    ax.legend(loc="upper right")
    ax.set_xlim(sim["x"].min(), sim["x"].max())
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _mass(label):
    digits = "".join(c for c in label if c.isdigit())
    return digits or label


def _elem(label):
    letters = "".join(c for c in label if c.isalpha())
    return letters or "S"


def _spin(I):
    twoI = round(2 * I)
    return f"{twoI}/2" if twoI % 2 else str(twoI // 2)


# --------------------------------------------------------------------------
# Web page
# --------------------------------------------------------------------------
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HFS Spectrum Simulator</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1d2733}
 header{background:#1f3a5f;color:#fff;padding:14px 22px}
 header h1{margin:0;font-size:19px;font-weight:600}
 header p{margin:4px 0 0;font-size:13px;opacity:.85}
 .wrap{display:flex;gap:20px;padding:20px;align-items:flex-start;flex-wrap:wrap}
 .panel{background:#fff;border:1px solid #dde2e8;border-radius:10px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .controls{width:330px;flex:0 0 auto}
 .output{flex:1;min-width:420px}
 h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:#5a6b7b;margin:0 0 12px}
 .grp{margin-bottom:16px}
 .grp h3{font-size:12px;color:#1f3a5f;margin:0 0 8px;border-bottom:1px solid #eef1f4;padding-bottom:4px}
 label{display:flex;justify-content:space-between;align-items:center;font-size:13px;margin:6px 0}
 label span{color:#3b4a59}
 input{width:120px;padding:5px 7px;border:1px solid #c7ced6;border-radius:6px;font-size:13px;text-align:right}
 button{width:100%;padding:10px;background:#1f3a5f;color:#fff;border:0;border-radius:8px;font-size:14px;
   font-weight:600;cursor:pointer;margin-top:6px}
 button:hover{background:#2b4d79}
 img{max-width:100%;border:1px solid #e3e7ec;border-radius:8px}
 table{border-collapse:collapse;width:100%;margin-top:12px;font-size:13px}
 td{padding:6px 8px;border-bottom:1px solid #eef1f4}
 td:first-child{color:#5a6b7b}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
 .meta{font-size:12px;color:#6b7886;margin-top:8px}
 .err{color:#b3261e;font-size:13px;margin-top:8px}
 .hint{font-size:11px;color:#90a0ad}
</style></head><body>
<header><h1>Hyperfine Spectrum Simulator</h1>
<p>RISE @ BECOLA &mdash; simulate an expected resonance-ionization spectrum and read off the statistical precision.</p></header>
<div class="wrap">
 <div class="panel controls">
  <div class="grp"><h3>Preset</h3>
   <label><span>Load isotope</span>
     <select id="preset" onchange="applyPreset()" style="width:120px;padding:5px;border:1px solid #c7ced6;border-radius:6px">
       <option value="">-- choose --</option>
     </select></label>
   <div class="hint">Fills spin, J, and A/B from the proposal presets.</div>
  </div>
  <div class="grp"><h3>Nucleus &amp; transition</h3>
   <label><span>Isotope label</span><input id="isotope" value="29S"></label>
   <label><span>Nuclear spin I</span><input id="spin" type="number" step="0.5" value="2.5"></label>
   <label><span>J lower</span><input id="J_lower" type="number" step="1" value="2"></label>
   <label><span>J upper</span><input id="J_upper" type="number" step="1" value="2"></label>
  </div>
  <div class="grp"><h3>Hyperfine constants (MHz)</h3>
   <label><span>A lower</span><input id="A_lower" type="number" value="120"></label>
   <label><span>A upper</span><input id="A_upper" type="number" value="45"></label>
   <label><span>B lower</span><input id="B_lower" type="number" value="-30"></label>
   <label><span>B upper</span><input id="B_upper" type="number" value="-12"></label>
   <div class="hint">Ignored for I = 0 (even-even). B is forced to 0 for I = 1/2.</div>
  </div>
  <div class="grp"><h3>Lineshape &amp; beam</h3>
   <label><span>Linewidth FWHM (MHz)</span><input id="linewidth" type="number" value="60"></label>
   <button id="offlineBtn" onclick="useOfflineLinewidth()" style="width:auto;padding:5px 10px;margin:2px 0 6px;font-size:12px;background:#3b5b86">Use offline linewidth</button>
   <label><span>Lorentzian fraction</span><input id="lorentz_fraction" type="number" step="0.05" value="0.3"></label>
   <label><span>Stopped-beam rate (pps)</span><input id="rate" type="number" value="120"></label>
   <label><span>Overall efficiency</span><input id="efficiency" type="number" step="0.0005" value="0.001"></label>
   <label><span>Background (cts/s)</span><input id="background" type="number" step="0.001" value="0.002"></label>
   <label><span>Measurement time (h)</span><input id="time_h" type="number" value="48"></label>
  </div>
  <div class="grp"><h3>Scan (blank = auto)</h3>
   <label><span>Half-range (MHz)</span><input id="scan_range" placeholder="auto"></label>
   <label><span>Steps</span><input id="n_steps" placeholder="auto"></label>
   <label><span>Random seed</span><input id="seed" type="number" value="1"></label>
  </div>
  <button onclick="simulate()">Simulate spectrum</button>
 </div>
 <div class="panel output">
  <h2>Simulated spectrum</h2>
  <div id="status" class="meta">Set parameters and click Simulate.</div>
  <img id="plot" style="display:none">
  <div id="err" class="err"></div>
  <table id="results" style="display:none"></table>
  <div id="meta" class="meta"></div>
  <div id="exports" style="display:none;margin-top:12px">
   <button onclick="downloadPNG()" style="width:auto;padding:7px 14px;margin-right:8px">Download PNG</button>
   <button onclick="downloadCSV()" style="width:auto;padding:7px 14px;background:#3b5b86">Download CSV</button>
  </div>
 </div>
</div>
<script>
const IDS=["isotope","spin","J_lower","J_upper","A_lower","A_upper","B_lower","B_upper",
 "linewidth","lorentz_fraction","rate","efficiency","background","time_h","scan_range","n_steps","seed"];
let PRESETS={}, CAL_FWHM=null, LAST=null;

async function loadPresets(){
 try{
  const d=await(await fetch("/api/presets")).json();
  PRESETS=d.presets||{}; CAL_FWHM=d.calibrated_linewidth_MHz;
  const sel=document.getElementById("preset");
  for(const name in PRESETS){
   const o=document.createElement("option");o.value=name;
   o.textContent=name+"  ("+(PRESETS[name].Jpi||"")+")";sel.appendChild(o);
  }
  const b=document.getElementById("offlineBtn");
  b.textContent=CAL_FWHM?("Use offline linewidth ("+CAL_FWHM+" MHz)"):"Offline linewidth unavailable";
  if(!CAL_FWHM)b.disabled=true;
 }catch(e){/* presets are optional */}
}
function applyPreset(){
 const name=document.getElementById("preset").value;if(!name||!PRESETS[name])return;
 const p=PRESETS[name];
 document.getElementById("isotope").value=p.isotope;
 document.getElementById("spin").value=p.spin;
 document.getElementById("J_lower").value=p.J_lower;
 document.getElementById("J_upper").value=p.J_upper;
 document.getElementById("A_lower").value=p.A_lower;
 document.getElementById("A_upper").value=p.A_upper;
 document.getElementById("B_lower").value=p.B_lower;
 document.getElementById("B_upper").value=p.B_upper;
}
function useOfflineLinewidth(){if(CAL_FWHM)document.getElementById("linewidth").value=CAL_FWHM;}

async function simulate(){
 const p={};
 for(const id of IDS){p[id]=document.getElementById(id).value;}
 document.getElementById("status").textContent="Simulating...";
 document.getElementById("err").textContent="";
 try{
  const r=await fetch("/api/simulate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
  const d=await r.json();
  if(d.error){document.getElementById("err").textContent=d.error;document.getElementById("status").textContent="Error.";return;}
  LAST=d;
  const img=document.getElementById("plot");img.src=d.image;img.style.display="block";
  document.getElementById("status").textContent=d.label+(d.even_even?" (even-even)":" (odd-A)");
  const t=document.getElementById("results");t.innerHTML="";
  if(d.fit_ok){for(const k in d.fit){t.innerHTML+="<tr><td>"+k+"</td><td>"+d.fit[k]+"</td></tr>";}}
  else{document.getElementById("err").textContent=d.fit_message||"Fit failed.";}
  t.style.display=d.fit_ok?"table":"none";
  document.getElementById("exports").style.display="block";
  document.getElementById("meta").textContent=
   "peak "+d.peak_counts+" cts | total "+d.total_counts+" cts | "+d.n_components+
   " component(s) | scan "+d.scan_range_MHz[0]+" .. "+d.scan_range_MHz[1]+" MHz, "+d.n_steps+" steps";
 }catch(e){document.getElementById("err").textContent=String(e);document.getElementById("status").textContent="Error.";}
}

function _dl(href,name){const a=document.createElement("a");a.href=href;a.download=name;
 document.body.appendChild(a);a.click();a.remove();}
function downloadPNG(){if(LAST)_dl(LAST.image,LAST.label+"_spectrum.png");}
function downloadCSV(){
 if(!LAST)return;
 let s="# isotope,"+LAST.label+"\\n# peak_counts,"+LAST.peak_counts+"\\n# total_counts,"+LAST.total_counts+"\\n";
 if(LAST.fit_ok){for(const k in LAST.fit){s+="# "+k.replace(/,/g," ")+","+LAST.fit[k]+"\\n";}}
 s+="frequency_MHz,counts\\n";
 for(let i=0;i<LAST.x.length;i++){s+=LAST.x[i]+","+LAST.counts[i]+"\\n";}
 _dl("data:text/csv;charset=utf-8,"+encodeURIComponent(s),LAST.label+"_spectrum.csv");
}
loadPresets();
</script>
</body></html>"""


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default logging
        return

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/presets":
            body = json.dumps({
                "presets": build_presets(),
                "calibrated_linewidth_MHz": get_calibrated_linewidth(),
            }).encode("utf-8")
            self._send(body, "application/json")
        else:
            self._send(b'{"error":"Not found."}', "application/json", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/simulate":
            self._send(b'{"error":"Not found."}', "application/json", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = run_simulation(payload)
            self._send(json.dumps(result).encode("utf-8"), "application/json")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            body = json.dumps({"error": f"{exc}"}).encode("utf-8")
            self._send(body, "application/json", 400)


def _free_port(preferred: int) -> int:
    for port in [preferred, 0]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            p = s.getsockname()[1]
            s.close()
            return p
        except OSError:
            continue
    return preferred


def run_server(host="127.0.0.1", port=8770, open_browser=True):
    port = _free_port(port)
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"HFS spectrum simulator running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description="Hyperfine spectrum simulator GUI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
