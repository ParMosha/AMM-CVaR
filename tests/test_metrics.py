"""Risk-metric tests: empirical CVaR must match the RU epigraph optimum."""

import numpy as np
import cvxpy as cp
import pytest

from amm_cvar.metrics import empirical_cvar, empirical_var, paired_comparison, summarize_losses


def ru_cvar_lp(losses, alpha):
    """Reference: CVaR by explicitly solving the Rockafellar-Uryasev LP."""
    x = np.asarray(losses, dtype=float)
    n = x.size
    t = cp.Variable()
    u = cp.Variable(n, nonneg=True)
    obj = t + (1.0 / ((1.0 - alpha) * n)) * cp.sum(u)
    prob = cp.Problem(cp.Minimize(obj), [u >= x - t])
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), float(t.value)


def test_cvar_matches_ru_epigraph():
    rng = np.random.default_rng(0)
    for alpha in (0.8, 0.9, 0.95, 0.99):
        for n in (50, 101, 250):
            x = rng.lognormal(mean=3.0, sigma=1.0, size=n)
            got = empirical_cvar(x, alpha)
            want, t_star = ru_cvar_lp(x, alpha)
            assert got == pytest.approx(want, rel=1e-4, abs=1e-6), (alpha, n, got, want)
            # VaR is the RU threshold (up to quantile convention)
            assert empirical_var(x, alpha) <= got + 1e-8


def test_cvar_known_sample():
    # losses sorted: 1..8, 80, 100.  At alpha=0.8 the tail is the worst 2.
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 80, 100], dtype=float)
    assert empirical_cvar(x, 0.8) == pytest.approx(90.0)
    # VaR convention: inf{t : P(L <= t) >= alpha}  ->  F(8) = 0.8
    assert empirical_var(x, 0.8) == pytest.approx(8.0)
    # RU threshold t* is any point of [8, 80]; the epigraph value is 90 either way
    for t in (8.0, 40.0, 80.0):
        phi = t + (1 / ((1 - 0.8) * 10)) * np.maximum(x - t, 0).sum()
        assert phi == pytest.approx(90.0)


def test_cvar_fractional_boundary():
    # (1-alpha) * n = 2.5  ->  half of the 3rd-worst counts
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0])
    # worst 2.5 of 10 at alpha=0.75: (30 + 20 + 0.5*10) / 2.5 = 22
    assert empirical_cvar(x, 0.75) == pytest.approx(22.0)


def test_cvar_monotone_in_alpha():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 20, size=400).clip(0)
    vals = [empirical_cvar(x, a) for a in (0.5, 0.8, 0.9, 0.95, 0.99)]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_cvar_bounded_by_max_and_ge_var():
    x = np.array([3.0, 5.0, 9.0, 12.0, 40.0, 4.0, 6.0, 7.0])
    for a in (0.5, 0.7, 0.9):
        c = empirical_cvar(x, a)
        v = empirical_var(x, a)
        assert v <= c + 1e-9 <= max(x) + 1e-9


def test_summarize_and_paired():
    a = np.array([1.0, 2.0, 3.0, 40.0])
    b = np.array([2.0, 3.0, 4.0, 10.0])
    s = summarize_losses(a, alphas=(0.75,))
    assert s["mean"] == pytest.approx(11.5)
    assert s["CVaR_0.75"] == pytest.approx(40.0)  # worst 1 of 4
    comp = paired_comparison(a, b, alpha=0.75)
    assert comp["win_rate_A"] == pytest.approx(0.75)  # a < b on 3 of 4
    assert comp["mean_diff_A_minus_B"] == pytest.approx((1 - 2 + 2 - 3 + 3 - 4 + 40 - 10) / 4)
