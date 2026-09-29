"""MagneTOF counting-chain corrections: echo counts and dead time.

The detection chain (MagneTOF -> amplifier -> discriminator -> TimeTagger4) was
characterized from same-bunch hit-pair intervals against an event-mixed
(different-bunch) baseline on eight pooled 32S scans (227k hits, March-June 2026;
regenerate with derive_counting_constants.py):

* Echo counts. About 10% of real hits are followed by a spurious second count
  70.5 ns later, with weaker structures at 60.5, 123, 131 and 142 ns (the last is
  the echo of the echo); together 11.8% of all hits. The echo delay and fraction
  are the same in every campaign, at any rate, for 32S and 34S.
* Dead time. Nothing is registered within 41 ns of a hit; from 45 to 66 ns about
  a third of second hits are still lost; the chain is fully live after ~67 ns.

Echoes are removed per bunch before ToF gating (a parent can sit just outside the
gate): a hit is an echo when its delay after ANY earlier hit in the same bunch lies
in ECHO_WINDOWS_NS. That cut also removes real ions landing in those windows, so
the windows are folded into the survival function used by the live-time
correction. Bunch intensities are strongly super-Poissonian (Fano factor 1.1-1.8),
so the dead-time loss is estimated from same-bunch hit pairs rather than from the
mean rate; see dwell_live_fractions.
"""

from __future__ import annotations

import numpy as np

# Echo delay windows [start, stop) in ns after an earlier hit in the same bunch.
ECHO_WINDOWS_NS: tuple[tuple[float, float], ...] = (
    (60.0, 61.5),
    (68.0, 72.0),
    (122.0, 124.0),
    (129.0, 133.0),
    (140.0, 144.0),
)

# Detector-only survival S_det(dt): the probability that an ion arriving dt after a
# registered (non-echo) hit in the same bunch is lost. Entry k covers [k, k+1) ns;
# zero beyond the table (51.4 ns integral; 66 ns with the echo veto). Measured
# 2026-09-28 by derive_counting_constants.py after echo removal, with the echo
# windows interpolated over (survival_function applies them separately) and a
# non-increasing (pool-adjacent-violators) fit, as a survival function must be.
DEADTIME_SURVIVAL_1NS = np.array(
    [1.0] * 41
    + [0.975, 0.759, 0.585, 0.423, 0.379, 0.377, 0.377, 0.377, 0.377, 0.370,
       0.370, 0.355, 0.355, 0.342, 0.342, 0.332, 0.332, 0.328, 0.328, 0.328,
       0.328, 0.328, 0.328, 0.313, 0.234, 0.187, 0.115, 0.030, 0.030, 0.030,
       0.030, 0.030, 0.030]
)


def tag_echo_hits(bunch_id, tof_s, windows_ns=ECHO_WINDOWS_NS) -> np.ndarray:
    """Return a mask (input order) of hits that are echoes of an earlier hit.

    A hit is an echo when its delay after any earlier hit in the same bunch,
    echoes included, falls in one of ``windows_ns``; chains (echo of an echo)
    are therefore caught by the primary window.
    """
    bunch_id = np.asarray(bunch_id)
    tof_ns = np.asarray(tof_s, dtype=float) * 1e9
    n = tof_ns.size
    echo = np.zeros(n, dtype=bool)
    if n < 2 or not windows_ns:
        return echo
    order = np.lexsort((tof_ns, bunch_id))
    b = bunch_id[order]
    t = tof_ns[order]
    flagged = np.zeros(n, dtype=bool)
    lag = 1
    while lag < n:
        same = b[lag:] == b[:-lag]
        if not same.any():
            break
        delay = t[lag:] - t[:-lag]
        in_window = np.zeros(n - lag, dtype=bool)
        for lo, hi in windows_ns:
            in_window |= (delay >= lo) & (delay < hi)
        flagged[lag:] |= same & in_window
        lag += 1
    echo[order] = flagged
    return echo


def survival_function(windows_ns=ECHO_WINDOWS_NS, step_ns: float = 1.0, max_ns: float = 200.0):
    """Total loss probability S(dt) on a grid of bin left edges (ns).

    Combines the detector dead time with the echo veto: an ion is lost if the
    chain is dead or the echo cut removes it, S = 1 - (1 - S_det)(1 - veto).
    """
    edges = np.arange(0.0, max_ns + step_ns, step_ns)
    left, right = edges[:-1], edges[1:]
    table_edges = np.arange(DEADTIME_SURVIVAL_1NS.size + 1, dtype=float)
    # Bin-averaged detector survival (exact for any step, table is piecewise constant).
    cum = np.concatenate([[0.0], np.cumsum(DEADTIME_SURVIVAL_1NS)])
    cum_at = lambda x: np.interp(x, table_edges, cum)
    s_det = (cum_at(right) - cum_at(left)) / step_ns
    veto = np.zeros_like(left)
    for lo, hi in windows_ns or ():
        overlap = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, None)
        veto = np.maximum(veto, overlap / step_ns)
    return left, 1.0 - (1.0 - s_det) * (1.0 - veto)


def effective_dead_time_ns(windows_ns=ECHO_WINDOWS_NS) -> float:
    """Integral of S(dt): the equivalent non-paralyzable dead time."""
    left, s = survival_function(windows_ns)
    step = left[1] - left[0] if left.size > 1 else 1.0
    return float(np.sum(s) * step)


def veto_function(windows_ns=ECHO_WINDOWS_NS, step_ns: float = 1.0, max_ns: float = 200.0):
    """Loss probability from the echo cut alone (applies after echo-tagged hits)."""
    edges = np.arange(0.0, max_ns + step_ns, step_ns)
    left, right = edges[:-1], edges[1:]
    veto = np.zeros_like(left)
    for lo, hi in windows_ns or ():
        overlap = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, None)
        veto = np.maximum(veto, overlap / step_ns)
    return left, veto


def dead_coverage_function(gate_tof_s, gate_us, windows_ns=ECHO_WINDOWS_NS, step_ns: float = 1.0, veto_only=False):
    """Return c(t): the fraction of the in-gate ToF density lost after a hit at t.

    ``gate_tof_s`` are the (echo-cleaned) in-gate hit times used as the ToF
    envelope. c(t) = sum_dt f_gate(t + dt) S(dt), zero once t + dt leaves the gate.
    With ``veto_only`` S is the echo cut alone: the loss behind an echo-tagged hit,
    which the chain does not go dead for but the tagger still vetoes after.
    """
    lo_ns, hi_ns = (float(v) * 1e3 for v in gate_us)
    tof_ns = np.asarray(gate_tof_s, dtype=float) * 1e9
    edges = np.arange(lo_ns, hi_ns + step_ns, step_ns)
    density, _ = np.histogram(tof_ns, edges)
    total = density.sum()
    if veto_only:
        s_left, s = veto_function(windows_ns, step_ns=step_ns)
    else:
        s_left, s = survival_function(windows_ns, step_ns=step_ns)
    n_s = s.size
    if total <= 0:
        return lambda t_s: np.zeros(np.shape(t_s), dtype=float)
    f = density / total
    # c on the grid of gate-bin left edges shifted back by the survival range.
    # Hit in grid bin m (left edge lo - n_s*step + m*step) kills ions in f bin m + d - n_s.
    n_f = f.size
    padded = np.concatenate([np.zeros(n_s), f, np.zeros(n_s)])
    coverage = np.zeros(n_f + n_s)
    for d in range(n_s):
        coverage += s[d] * padded[d: d + n_f + n_s]
    grid_centers = lo_ns - n_s * step_ns + (np.arange(n_f + n_s) + 0.5) * step_ns

    def c_of_t(t_s):
        return np.interp(np.asarray(t_s, dtype=float) * 1e9, grid_centers, coverage, left=0.0, right=0.0)

    return c_of_t


def dwell_live_fractions(
    hit_bunch_code,
    hit_tof_s,
    hit_in_gate,
    bunch_dwell_code,
    n_dwells: int,
    coverage,
    echo_bunch_code=None,
    echo_tof_s=None,
    veto_coverage=None,
):
    """Live fraction of in-gate ions per dwell, estimated bunch by bunch.

    Every in-gate ion lost in a bunch sits in the dead window of exactly one
    registered hit j, which covers the fraction c(t_j) of the in-gate density.
    With T true and R registered in-gate ions in the bunch,
    T - R = sum_j c(t_j) (T - [j in gate]), so
    T = (R - sum_{j in gate} c(t_j)) / (1 - sum_j c(t_j)). Solving per bunch keeps
    bunch-intensity fluctuations and the higher orders of the loss. Echo-tagged
    hits add only their veto coverage. A Monte Carlo of the measured chain
    (tests/) recovers the true in-gate count to ~1% at 4 ions/bunch with 40%
    intensity variance, where the raw loss is 25%. L = sum R / sum T per dwell.

    ``hit_bunch_code`` indexes ``bunch_dwell_code`` (one entry per bunch); hits
    must be echo-cleaned. Returns (live[n_dwells], true_in_gate[n_dwells],
    registered_in_gate[n_dwells]).
    """
    hit_bunch_code = np.asarray(hit_bunch_code, dtype=np.int64)
    in_gate = np.asarray(hit_in_gate, dtype=bool)
    n_bunches = np.asarray(bunch_dwell_code).size
    c = coverage(hit_tof_s)
    n_gate_b = np.bincount(hit_bunch_code, weights=in_gate.astype(float), minlength=n_bunches)
    c_all_b = np.bincount(hit_bunch_code, weights=c, minlength=n_bunches)
    if echo_bunch_code is not None and veto_coverage is not None and len(echo_bunch_code):
        c_all_b += np.bincount(np.asarray(echo_bunch_code, dtype=np.int64), weights=veto_coverage(echo_tof_s),
                               minlength=n_bunches)
    c_gate_b = np.bincount(hit_bunch_code, weights=c * in_gate, minlength=n_bunches)
    true_b = np.maximum(n_gate_b - c_gate_b, 0.0) / np.clip(1.0 - c_all_b, 0.05, None)
    dwell = np.asarray(bunch_dwell_code, dtype=np.int64)
    true_d = np.bincount(dwell, weights=true_b, minlength=n_dwells)
    n_gate_d = np.bincount(dwell, weights=n_gate_b, minlength=n_dwells)
    with np.errstate(divide="ignore", invalid="ignore"):
        live = np.where(true_d > 0, n_gate_d / true_d, 1.0)
    return np.clip(live, 0.05, 1.0), true_d, n_gate_d
