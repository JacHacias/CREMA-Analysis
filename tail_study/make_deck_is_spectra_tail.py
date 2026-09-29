"""DNP deck slide 6/14 spectra with the adopted line shape (energy-loss tail, +190.8 V calibration).

Same figure as DNP2026_talk/figs/is_spectra_v2.png (made by make_is_spectra_v2.py in session
edde5d18), from the 2026-03-23 32S/34S pair, replayed read-only with the tail model: the dashed
lines are the no-loss core centers, whose difference is the isotope shift. Writes
DNP2026_talk/figs/is_spectra_v3.png. Run from the repo root with the repo .venv.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path('C:/Users/EMALAB/Documents/Jackson/CREMA-Analysis')
sys.path.insert(0, str(REPO))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import rate_spectrum  # noqa: E402
import reanalyze_library as rl  # noqa: E402

DEFAULTS = json.loads((REPO / 'hfs_gui/analysis_defaults.json').read_text(encoding='utf-8'))
OUT = 'C:/Users/EMALAB/Documents/Jackson/DNP2026_talk/figs/is_spectra_v3.png'
RUN = 'sulfur_2026-03-23'
captured = []
_orig = rate_spectrum.two_isotope_rate_fit


def _capture(*args, **kwargs):
    res = _orig(*args, **kwargs)
    captured.append(res)
    return res


rate_spectrum.two_isotope_rate_fit = _capture
rows = list(csv.DictReader(open(REPO / 'hfs_gui/data_library/isotope_shift_library.csv', encoding='utf-8')))
row = next(r for r in rows if r['run_label'] == RUN and r['comparison'] == '34S-32S')
overrides = {'bootstrap_replicas': 0, 'tail_model': DEFAULTS['tail_model'], 'shape_transfer': DEFAULTS['shape_transfer'],
             'voltage_offset_V': DEFAULTS['voltage_offset_V']}
with tempfile.TemporaryDirectory() as tmp:
    new = rl.replay_row(row, overrides, Path(tmp))
plt.close('all')
print('library row:', row['isotope_shift_MHz'], '| replayed (no bootstrap):', new['isotope_shift_MHz'])
res = captured[-1]
nu0 = res['nu0_GHz']
r32, r34 = res['results']['32S'], res['results']['34S']
shift = res['isotope_shift_GHz'] * 1000.0
print(f'captured: shift {shift:.2f} MHz, cores {res["center1_GHz"] * 1e3:.2f} / {res["center2_GHz"] * 1e3:.2f} MHz')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.9, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
COL = {'32S': ('#3b6fb6', '#1f4f94'), '34S': ('#e08a2e', '#b8651a')}
fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5 / 1.4853), dpi=300, sharex=True)
centers = {}
for ax, lab, r in zip(axes, ('32S', '34S'), (r32, r34)):
    disp = r['display']
    off = (r['nu_ref_GHz'] - nu0) * 1000.0
    ok = np.isfinite(disp['rate'])
    c, dark = COL[lab]
    ax.errorbar(disp['centers_MHz'][ok] + off, disp['rate'][ok], yerr=disp['rate_err'][ok], fmt='o', ms=2.6,
                color=c, ecolor=c, elinewidth=0.8, alpha=0.85, zorder=2)
    ax.plot(disp['curve_x_MHz'] + off, disp['curve_y'], color=dark, lw=1.5, zorder=3)
    centers[lab] = (r['center_abs_GHz'] - nu0) * 1000.0
    top = float(np.nanmax(disp['rate'][ok] + disp['rate_err'][ok]))
    ax.set_ylim(0, top * (1.42 if lab == '32S' else 1.18))
    ax.text(0.025, 0.93, f'$^{{\\mathbf{{{lab[:2]}}}}}$S', transform=ax.transAxes, ha='left', va='top',
            fontsize=12, fontweight='bold', color=dark)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
for ax in axes:
    for lab, x in centers.items():
        ax.axvline(x, color=COL[lab][1], ls=(0, (4, 2.5)), lw=1.0, zorder=1)
ax0 = axes[0]
y_arrow = ax0.get_ylim()[1] * 0.83
ax0.annotate('', xy=(centers['34S'], y_arrow), xytext=(centers['32S'], y_arrow),
             arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.1, shrinkA=0, shrinkB=0))
ax0.text(0.5 * (centers['32S'] + centers['34S']), y_arrow * 1.03, f'{shift:.0f} MHz', ha='center', va='bottom',
         fontsize=10.5, fontweight='bold', color='0.15')
axes[1].set_xlim(-1100, 1250)
axes[1].set_xlabel('Rest-frame frequency offset (MHz)', fontsize=10)
fig.supylabel('Ions per bunch', fontsize=10, x=0.012)
fig.tight_layout(h_pad=0.4)
fig.subplots_adjust(left=0.115)
fig.savefig(OUT, facecolor='white')
print('saved', OUT)
