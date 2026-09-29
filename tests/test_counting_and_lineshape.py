"""Checks for the v2 counting corrections, ripple lineshape and Poisson fit.

Run from the repo root:  .venv\\Scripts\\python.exe -m unittest discover -s tests
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import counting_corrections as cc  # noqa: E402
import rate_spectrum as rs  # noqa: E402
import ripple_lineshape as rl  # noqa: E402

GATE_US = (4.25, 5.5)
ECHO_DELAY_NS = 70.5


def _dead_time_sampler(rng):
    """Draw dead times whose survival P(tau > dt) is the measured S_det table."""
    table = cc.DEADTIME_SURVIVAL_1NS
    cdf = 1.0 - table                      # P(tau <= k) on 1 ns steps
    def draw(size):
        u = rng.random(size)
        k = np.searchsorted(cdf, u)        # first k with cdf >= u
        return (k.astype(float) + rng.random(size)) if size else np.empty(0)
    return draw


def simulate_detector(rng, n_bunches, mu, *, rel_var=0.4, p_echo=0.10, tof_mean=4.9e-6, tof_sd=0.3e-6):
    """Bunches with gamma-distributed intensity through the dead time and echo chain.

    Returns (bunch_id, tof_s, is_true, n_true_in_gate): registered hits (echoes
    included, as the TimeTagger records them) and the true in-gate ion count.
    Echoes register only when the chain is live and do not start a dead time,
    matching the flat 75-110 ns pair efficiency seen in the data.
    """
    draw_tau = _dead_time_sampler(rng)
    shape = 1.0 / rel_var
    intensity = rng.gamma(shape, mu / shape, n_bunches)
    counts = rng.poisson(intensity)
    bunch_ids, tofs, truth = [], [], []
    n_true_gate = 0
    for b, k in enumerate(counts):
        if k == 0:
            continue
        t = np.sort(np.clip(rng.normal(tof_mean, tof_sd, k), 4.0e-6, 6.0e-6)) * 1e9
        n_true_gate += int(np.sum((t > GATE_US[0] * 1e3) & (t < GATE_US[1] * 1e3)))
        events = [(ti, True) for ti in t]
        events.sort()
        dead_until = -np.inf
        queue = list(events)
        i = 0
        while i < len(queue):
            ti, real = queue[i]
            i += 1
            if ti < dead_until:
                continue
            bunch_ids.append(b)
            tofs.append(ti * 1e-9)
            truth.append(real)
            if real:
                dead_until = ti + float(draw_tau(1)[0])
                if rng.random() < p_echo:
                    queue.append((ti + ECHO_DELAY_NS, False))
                    queue.sort(key=lambda e: e[0])
    return np.array(bunch_ids), np.array(tofs), np.array(truth, dtype=bool), n_true_gate


class CountingCorrectionTests(unittest.TestCase):
    def test_echo_tagging_finds_simulated_echoes(self):
        rng = np.random.default_rng(1)
        bunch, tof, real, _ = simulate_detector(rng, 20000, 1.0)
        echo = cc.tag_echo_hits(bunch, tof)
        self.assertGreater(np.mean(echo[~real]), 0.99)          # echoes found
        self.assertLess(np.mean(echo[real]), 0.02)              # few real ions vetoed at this rate

    def test_live_time_correction_recovers_true_counts(self):
        # Rates spanning wing to the highest observed 32S peak (~3.5 ions/bunch).
        for mu in (0.2, 1.0, 2.5, 4.0):
            rng = np.random.default_rng(int(mu * 10))
            n_bunches = 6000
            bunch, tof, real, n_true = simulate_detector(rng, n_bunches, mu)
            echo = cc.tag_echo_hits(bunch, tof)
            clean = ~echo
            tof_us = tof * 1e6
            in_gate = clean & (tof_us > GATE_US[0]) & (tof_us < GATE_US[1])
            coverage = cc.dead_coverage_function(tof[in_gate], GATE_US)
            veto = cc.dead_coverage_function(tof[in_gate], GATE_US, veto_only=True)
            live, _, n_gate = cc.dwell_live_fractions(
                bunch[clean], tof[clean], in_gate[clean], np.zeros(n_bunches, dtype=int), 1, coverage,
                echo_bunch_code=bunch[echo], echo_tof_s=tof[echo], veto_coverage=veto,
            )
            corrected = n_gate[0] / live[0]
            raw_bias = n_gate[0] / n_true - 1.0
            bias = corrected / n_true - 1.0
            with self.subTest(mu=mu):
                self.assertLess(abs(bias), 0.012 + 3.0 / math.sqrt(n_true), f"raw {raw_bias:+.3f} corrected {bias:+.3f}")

    def test_survival_includes_veto_windows(self):
        left, s = cc.survival_function()
        self.assertTrue(np.all(s[:41] > 0.97))
        for lo, hi in cc.ECHO_WINDOWS_NS:
            inside = (left >= lo) & (left + 1.0 <= hi)
            self.assertTrue(np.allclose(s[inside], 1.0))
        self.assertAlmostEqual(cc.effective_dead_time_ns(), 66.0, delta=2.0)


class RippleLineshapeTests(unittest.TestCase):
    def test_profile_normalized_and_arcsine_variance(self):
        x = np.linspace(-3000.0, 3000.0, 60001)
        a, sd = 138.8, 40.0
        p = rl.ripple_voigt(x, sd, 0.0, a, laser_fwhm_MHz=0.0)
        self.assertAlmostEqual(np.trapezoid(p, x), 1.0, places=4)
        var = np.trapezoid(p * x ** 2, x)
        self.assertAlmostEqual(var, sd ** 2 + a ** 2 / 2.0, delta=0.5)   # arcsine variance a^2/2

    def test_laser_width_adds_in_quadrature(self):
        sigma, gamma = rl.instrument_widths(0.0, 0.0, 15.0 * math.sqrt(2.0), "gaussian")
        self.assertAlmostEqual(sigma * rl.FWHM_PER_SIGMA, 15.0 * math.sqrt(2.0), places=9)
        self.assertEqual(gamma, 0.0)

    def test_ripple_halfwidth_32S(self):
        a = rl.ripple_halfwidth_MHz(756.36e6, 31.972071, 10102.7, 4.5)
        self.assertAlmostEqual(a, 138.8, delta=0.5)


class PoissonFitTests(unittest.TestCase):
    def test_fit_recovers_center_with_uneven_exposure(self):
        rng = np.random.default_rng(7)
        x = np.repeat(np.arange(-700.0, 700.0, 30.0), 3) + rng.normal(0, 4, 141)
        expo = rng.integers(150, 600, x.size).astype(float)       # uneven dwell per setpoint
        truth = dict(amp=0.8 * 400, x0=23.0, sd=70.0, g=20.0, b0=0.05)
        a = 138.8
        rate = truth["b0"] + truth["amp"] * rl.ripple_voigt(x - truth["x0"], truth["sd"], truth["g"], a)
        centers = []
        for _ in range(12):
            n = rng.poisson(expo * rate)
            fit = rs.fit_poisson_lineshape(x, n, expo, ripple_halfwidth=a, laser_fwhm_MHz=rl.LASER_FWHM_MHZ)
            centers.append((fit.params["center"], fit.errors["center"]))
        values = np.array([c for c, _ in centers])
        errors = np.array([e for _, e in centers])
        pulls = (values - truth["x0"]) / errors
        self.assertLess(abs(np.mean(values) - truth["x0"]), 3.0 * np.mean(errors) / math.sqrt(values.size))
        self.assertLess(abs(np.std(pulls) - 1.0), 0.45)


class TailLineshapeTests(unittest.TestCase):
    def test_fft_profile_matches_direct_voigt_without_tail(self):
        x = np.linspace(-850.0, 700.0, 500) + 0.37
        grid = rs.ProfileGrid(x)
        direct = rl.ripple_voigt(x - 35.0, 110.0, 10.0, 138.8, laser_fwhm_MHz=rl.LASER_FWHM_MHZ)
        fft = grid.profile(x, 35.0, 110.0, 10.0, 138.8, 0.0, 100.0, laser_fwhm_MHz=rl.LASER_FWHM_MHZ)
        self.assertLess(np.max(np.abs(fft - direct)) / direct.max(), 1e-4)

    def test_tail_normalized_one_sided_and_shifts_mean(self):
        x = np.arange(-6000.0, 3000.0, 0.5)
        f, lam = 0.4, 250.0
        base = dict(center=0.0, sigma_doppler=80.0, gamma=5.0, ripple_halfwidth=138.8, tail_fraction=f, tail_length=lam)
        low = rs.line_profile(x, base, laser_fwhm_MHz=rl.LASER_FWHM_MHZ, tail_model="exponential", tail_side=-1)
        high = rs.line_profile(x, base, laser_fwhm_MHz=rl.LASER_FWHM_MHZ, tail_model="exponential", tail_side=+1)
        self.assertAlmostEqual(np.trapezoid(low, x), 1.0, delta=2e-3)   # Lorentzian wings beyond the range
        # the exponential moves the mean by -f*lambda (Lorentzian wings cut symmetrically)
        m = np.abs(x) < 2500
        self.assertAlmostEqual(np.trapezoid(low[m] * x[m], x[m]) / np.trapezoid(low[m], x[m]), -f * lam, delta=6.0)
        self.assertAlmostEqual(np.trapezoid(high[m] * x[m], x[m]) / np.trapezoid(high[m], x[m]), +f * lam, delta=6.0)

    def test_fit_recovers_injected_tail_and_core_center(self):
        rng = np.random.default_rng(11)
        x = np.repeat(np.arange(-850.0, 700.0, 10.0), 2) + rng.normal(0, 2, 310)
        expo = rng.integers(300, 900, x.size).astype(float)
        truth = dict(center=60.0, sigma_doppler=90.0, gamma=10.0, ripple_halfwidth=138.8, tail_fraction=0.5, tail_length=260.0)
        rate = 0.05 + 1.5 * 400 * rs.line_profile(x, truth, laser_fwhm_MHz=rl.LASER_FWHM_MHZ, tail_model="exponential")
        n = rng.poisson(expo * rate)
        fit = rs.fit_poisson_lineshape(x, n, expo, ripple_halfwidth=138.8, laser_fwhm_MHz=rl.LASER_FWHM_MHZ,
                                       tail_model="exponential", tail_side=-1)
        p, e = fit.params, fit.errors
        for key in ("center", "tail_fraction", "tail_length", "sigma_doppler"):
            self.assertLess(abs(p[key] - truth[key]), 3.5 * e[key] + 1e-9, msg=f"{key}: {p[key]} vs {truth[key]} +/- {e[key]}")
        self.assertGreater(fit.deviance_symmetric - fit.deviance, 25.0)   # the tail is detected
        # tail fixed at the truth (shape transfer): center error shrinks
        fixed = rs.fit_poisson_lineshape(x, n, expo, ripple_halfwidth=138.8, laser_fwhm_MHz=rl.LASER_FWHM_MHZ,
                                         tail_model="exponential", fixed_tail_fraction=0.5, fixed_tail_length=260.0)
        self.assertLess(fixed.errors["center"], e["center"])
        self.assertLess(abs(fixed.params["center"] - truth["center"]), 3.5 * fixed.errors["center"])


if __name__ == "__main__":
    unittest.main()
