"""Measure the MagneTOF echo structure and dead-time survival from raw scans.

Same-bunch hit-pair intervals are compared with an event-mixed baseline (hits of
bunch r paired with bunch r + MIX_LAG, same dwell in practice), normalized on the
150-400 ns plateau where the chain is fully live. Prints:

1. significant echo structures in the raw data (excess over the baseline),
2. the echo-tagged fraction per file with counting_corrections.ECHO_WINDOWS_NS,
3. the detector-only survival table S_det(dt) at 1 ns after echo removal, as a
   literal for counting_corrections.DEADTIME_SURVIVAL_1NS.

Run: python derive_counting_constants.py [scan.csv ...]   (default: pooled 32S set)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import counting_corrections as cc

DAQ = Path(r"C:\Users\EMALAB\Desktop\DBD_daq_emalab\data")
MARCH = Path(r"C:\Users\EMALAB\Documents\Jackson\Archived\data\S data for analysis")
DEFAULT_FILES = [
    DAQ / "scan_20260508_131731.csv",
    DAQ / "scan_20260512_144505.csv",
    DAQ / "scan_20260511_083642.csv",
    DAQ / "scan_20260601_120201.csv",
    DAQ / "scan_20260505_194044.csv",
    MARCH / "32S_3-23-26.csv",
    MARCH / "32S_3-27-26.csv",
    MARCH / "32S_3-24-26.csv",
]
MIX_LAG = 7
PLATEAU_NS = (150.0, 400.0)
MAX_NS = 600.0


def load_hits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["channel", "tof", "bunch_id"])
    hits = frame[(frame.channel == 2) & (frame.tof > 0)]
    return hits.tof.to_numpy() * 1e9, hits.bunch_id.to_numpy()


def _sorted_bunches(t_ns: np.ndarray, bunch: np.ndarray):
    order = np.lexsort((t_ns, bunch))
    t, b = t_ns[order], bunch[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    counts = np.diff(np.r_[starts, b.size])
    return t, starts, counts


def pair_histograms(t_ns: np.ndarray, bunch: np.ndarray, edges: np.ndarray):
    """(same-bunch, event-mixed) interval histograms."""
    t, starts, counts = _sorted_bunches(t_ns, bunch)
    same = np.zeros(edges.size - 1)
    b_of_hit = np.repeat(np.arange(starts.size), counts)
    for lag in range(1, int(counts.max(initial=1))):
        ok = b_of_hit[lag:] == b_of_hit[:-lag]
        same += np.histogram((t[lag:] - t[:-lag])[ok], edges)[0]
    mixed = np.zeros(edges.size - 1)
    first, second = np.arange(starts.size - MIX_LAG), np.arange(MIX_LAG, starts.size)
    for a in range(int(counts.max(initial=1))):
        for c in range(int(counts.max(initial=1))):
            sel = (counts[first] > a) & (counts[second] > c)
            if not sel.any():
                continue
            d = np.abs(t[starts[second[sel]] + c] - t[starts[first[sel]] + a])
            mixed += np.histogram(d, edges)[0]
    return same, mixed


def pooled(files, echo_windows=None):
    edges = np.arange(0.0, MAX_NS + 1.0, 1.0)
    left = edges[:-1]
    plateau = (left >= PLATEAU_NS[0]) & (left < PLATEAU_NS[1])
    same_total = np.zeros(left.size)
    expected_total = np.zeros(left.size)
    n_hits = 0
    per_file = []
    for path in files:
        t, b = load_hits(path)
        keep = np.ones(t.size, dtype=bool)
        if echo_windows is not None:
            keep = ~cc.tag_echo_hits(b, t * 1e-9, echo_windows)
            per_file.append((Path(path).name, t.size, 1.0 - keep.mean()))
        same, mixed = pair_histograms(t[keep], b[keep], edges)
        same_total += same
        expected_total += mixed * same[plateau].sum() / max(mixed[plateau].sum(), 1.0)
        n_hits += t.size
    return left, same_total, expected_total, n_hits, per_file


def echo_structures(left, same, expected, n_hits, z_cut=4.0, start_ns=45.0):
    z = (same - expected) / np.sqrt(np.clip(expected, 1.0, None))
    idx = np.flatnonzero((z > z_cut) & (left >= start_ns))
    clusters: list[list[int]] = []
    for i in idx:
        if clusters and i - clusters[-1][-1] <= 2:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    for c in clusters:
        excess = float((same[c[0]:c[-1] + 1] - expected[c[0]:c[-1] + 1]).sum())
        print(f"  {left[c[0]]:5.0f}-{left[c[-1]] + 1:5.0f} ns  excess {excess:8.0f}  = {excess / n_hits:.3%} of hits")


def nonincreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit (pool adjacent violators): a survival function."""
    blocks: list[list[float]] = []
    for v in values:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([[v] * int(n) for v, n in blocks])


def detector_survival_table(left, same, expected, windows, smooth_from=45, smooth_to=67, tail_cut=0.03):
    eps = same / np.where(expected > 0, expected, np.nan)
    raw = 1.0 - eps
    in_window = np.zeros(left.size, dtype=bool)
    for lo, hi in windows:
        in_window |= (left + 1.0 > lo) & (left < hi)
    s = raw.copy()
    good = ~in_window & np.isfinite(s)
    s[~good] = np.interp(left[~good], left[good], s[good])
    smoothed = s.copy()
    for k in range(smooth_from, left.size):
        lo, hi = max(k - 2, smooth_from), k + 3
        seg = s[lo:hi][good[lo:hi]] if good[lo:hi].any() else s[lo:hi]
        smoothed[k] = float(np.mean(seg))
    smoothed[:smooth_from] = s[:smooth_from]
    smoothed = np.clip(smoothed, 0.0, 1.0)
    end = smooth_to
    while end < left.size and smoothed[end:end + 5].max(initial=0.0) > tail_cut:
        end += 1
    table = nonincreasing(smoothed[:end])
    table[: int(np.argmax(table < 0.99))] = 1.0
    return table


def main(argv=None) -> int:
    files = [Path(p) for p in (argv if argv else [])] or DEFAULT_FILES
    print(f"files: {len(files)}")
    left, same, expected, n_hits, _ = pooled(files)
    print(f"raw pooled hits: {n_hits}\nsignificant echo structures (raw, z > 4):")
    echo_structures(left, same, expected, n_hits)
    left, same, expected, _, per_file = pooled(files, echo_windows=cc.ECHO_WINDOWS_NS)
    print("\necho-tagged fraction with ECHO_WINDOWS_NS:")
    for name, n, frac in per_file:
        print(f"  {name:28s} hits {n:6d}  tagged {frac:6.2%}")
    table = detector_survival_table(left, same, expected, cc.ECHO_WINDOWS_NS)
    print(f"\nDEADTIME_SURVIVAL_1NS ({table.size} entries, sum {table.sum():.1f} ns):")
    print("[" + ", ".join(f"{v:.3f}" for v in table) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
