"""Cost-model tests: convexity, xy=k derivation sanity, impact regression."""

import numpy as np
import pytest

from amm_cvar.cost_model import (
    fit_impact_regression,
    quadratic_impact,
    scenario_cost_numpy,
    true_execution_cost,
    xyk_exact_impact,
)


def test_xyk_impact_convex_and_matches_taylor():
    L = 5e6
    n = np.array([0.0, 1e3, 5e4, 2e5, 1e6])
    c = xyk_exact_impact(n, L)
    # second-order Taylor around 0 is n^2 / L
    assert c[1] == pytest.approx(n[1] ** 2 / (L + n[1]))
    # convexity along the grid (second differences >= 0)
    d2 = np.diff(c, 2)
    assert np.all(d2 >= -1e-6)
    # for small n the quadratic surrogate is accurate
    small = 1e-2 * L
    rel = abs(quadratic_impact(small, L) - xyk_exact_impact(small, L)) / xyk_exact_impact(small, L)
    assert rel < 0.02


def test_quadratic_impact_scales_quadratically():
    L = np.array([1e6, 2e6])
    f = np.array([1e4, 2e4])
    c1 = quadratic_impact(f, L)
    f2 = 2 * f
    assert quadratic_impact(f2, L) == pytest.approx(4 * c1)


def test_scenario_cost_shape_and_finiteness():
    rng = np.random.default_rng(0)
    M, K = 4, 7
    f = rng.uniform(1e3, 1e5, size=M)
    L_scen = rng.uniform(1e6, 1e7, size=(M, K))
    a = np.full(M, 0.003)
    b = np.ones(M)
    gas = np.full(M, 5.0)
    costs = scenario_cost_numpy(f, L_scen, a, b, gas=gas)
    assert costs.shape == (K,)
    assert np.all(np.isfinite(costs))
    # worse liquidity -> higher cost
    worse = scenario_cost_numpy(f, L_scen * 0.5, a, b, gas=gas)
    assert np.all(worse > costs)


def test_scenario_cost_convexity_midpoint():
    rng = np.random.default_rng(2)
    M, K = 3, 5
    L_scen = rng.uniform(1e6, 5e6, size=(M, K))
    a = np.full(M, 0.001)
    b = np.ones(M)
    for _ in range(20):
        f1 = rng.uniform(0, 2e5, size=M)
        f2 = rng.uniform(0, 2e5, size=M)
        mid = 0.5 * (f1 + f2)
        c1 = scenario_cost_numpy(f1, L_scen, a, b)
        c2 = scenario_cost_numpy(f2, L_scen, a, b)
        cm = scenario_cost_numpy(mid, L_scen, a, b)
        assert np.all(cm <= 0.5 * (c1 + c2) + 1e-8)


def test_true_execution_cost_charges_fixed_gas_once_per_pool():
    f = np.array([1e5, 0.0, 5e4])
    L = np.array([1e7, 1e7, 1e7])
    a = np.array([0.003, 0.003, 0.003])
    b = np.ones(3)
    gas = np.array([8.0, 8.0, 8.0])
    c = true_execution_cost(f, L, a, b, gas)
    fee = 0.003 * 1.5e5
    impact = 1e5**2 / 1e7 + 5e4**2 / 1e7
    assert c == pytest.approx(fee + impact + 16.0)


def test_fit_impact_regression_recovers_quadratic():
    rng = np.random.default_rng(3)
    n = rng.uniform(2e3, 5e5, size=20000)
    a_true, b_true = 1e-3, 2e-9
    y = a_true * n + b_true * n**2 + rng.normal(0, 30.0, size=n.size)
    df = {"notional": n, "cost_usd": y}
    fit = fit_impact_regression(df)
    assert fit["b_tape"] == pytest.approx(b_true, rel=0.25)
    assert fit["a_tape"] == pytest.approx(a_true, rel=0.25)
    assert fit["r2"] > 0.9
    assert fit["b_tape"] > 0
