"""Optimizer tests: DCP compliance, feasibility, dominance over baselines."""

import numpy as np
import cvxpy as cp
import pytest

from amm_cvar.baselines import mean_only_allocation, single_pool_best_mean
from amm_cvar.cost_model import scenario_cost_numpy
from amm_cvar.metrics import empirical_cvar
from amm_cvar.optimizer import problem_dimensions, solve_cvar_routing
from amm_cvar.scenarios import scenario_cost_matrix


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(7)
    M, K = 5, 40
    L_nom = rng.uniform(5e6, 5e7, size=M)
    a = np.array([0.0005, 0.003, 0.003, 0.0001, 0.003])
    b = np.ones(M)
    gas = np.array([6.0, 5.0, 5.0, 7.0, 5.5])
    xi = np.sort(rng.beta(2, 5, size=K))
    deltas = np.outer(rng.uniform(0.05, 0.4, size=M), xi)
    L_scen = L_nom[:, None] * (1 - np.clip(deltas, 0, 0.6))
    return {"M": M, "K": K, "L_scen": L_scen, "a": a, "b": b, "gas": gas, "F": 2.5e5}


def test_formulation_is_dcp(small_problem):
    p = small_problem
    M, K = p["M"], p["K"]
    f = cp.Variable(M, nonneg=True)
    z = cp.Variable(M, nonneg=True)
    t = cp.Variable()
    u = cp.Variable(K, nonneg=True)
    Ck = []
    for k in range(K):
        slip = cp.sum(cp.hstack([p["b"][m] * cp.quad_over_lin(f[m], p["L_scen"][m, k]) for m in range(M)]))
        Ck.append(p["a"] @ f + slip + p["gas"] @ z)
    obj = (1 / K) * cp.sum(cp.hstack(Ck)) + (t + (1 / ((1 - 0.95) * K)) * cp.sum(u))
    cons = [cp.sum(f) == p["F"], z <= 1, f <= p["F"] * z, u >= cp.hstack(Ck) - t]
    prob = cp.Problem(cp.Minimize(obj), cons)
    assert prob.is_dcp(), "the assignment formulation must be DCP"


def test_solution_respects_constraints(small_problem):
    p = small_problem
    res = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"])
    f, z = res["allocation"], res["usage"]
    assert res["status"] in ("optimal", "optimal_inaccurate")
    assert f.sum() == pytest.approx(p["F"], rel=1e-5)
    assert np.all(f >= -1e-6)
    assert np.all(z >= -1e-6) and np.all(z <= 1 + 1e-6)
    assert np.all(f <= p["F"] * z + 1e-4)


def test_usage_collapses_to_allocation_share(small_problem):
    """Collapse lemma: with a linear penalty in z the optimum has z_m = f_m / F."""
    p = small_problem
    res = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"], eta=1.0)
    assert res["usage"] == pytest.approx(res["allocation"] / p["F"], abs=1e-4)


def test_reproducible(small_problem):
    p = small_problem
    r1 = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"])
    r2 = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"])
    assert r1["allocation"] == pytest.approx(r2["allocation"], rel=1e-4, abs=1e-4)


def test_mean_only_dominates_single_pool_on_mean(small_problem):
    p = small_problem
    mo = mean_only_allocation(p["F"], p["L_scen"], p["a"], p["b"], p["gas"])
    sp = single_pool_best_mean(p["F"], p["L_scen"], p["a"], p["b"], p["gas"])
    assert mo["mean_cost"] <= sp["mean_cost"] + 1e-5


def test_cvar_solution_has_lower_cvar_than_mean_only(small_problem):
    p = small_problem
    mo = mean_only_allocation(p["F"], p["L_scen"], p["a"], p["b"], p["gas"])
    pure = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"], alpha=0.95, lam=1e6)
    cvar_mo = empirical_cvar(mo["scenario_costs"], 0.95)
    cvar_pure = empirical_cvar(pure["scenario_costs"], 0.95)
    assert cvar_pure <= cvar_mo + 1e-4


def test_lambda_zero_matches_mean_only(small_problem):
    p = small_problem
    r = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"], lam=0.0)
    mo = mean_only_allocation(p["F"], p["L_scen"], p["a"], p["b"], p["gas"])
    assert r["allocation"] == pytest.approx(mo["allocation"], rel=1e-3, abs=1e-2)


def test_dimensions_count(small_problem):
    d = problem_dimensions(5, 40)
    assert d["n_variables"] == 5 + 5 + 1 + 40
    assert d["n_constraints"] == 1 + 5 + 5 + 40 + 40


def test_invalid_inputs_raise(small_problem):
    p = small_problem
    with pytest.raises(ValueError):
        solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], 0.0)
    with pytest.raises(ValueError):
        solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"], alpha=1.0)
    with pytest.raises(ValueError):
        solve_cvar_routing(-p["L_scen"], p["a"], p["b"], p["gas"], p["F"])


def test_higher_alpha_focuses_allocation_on_safer_pools(small_problem):
    """Risk-averse solutions must not do worse on the tail than risk-neutral ones."""
    p = small_problem
    costs = {}
    for al in (0.8, 0.99):
        res = solve_cvar_routing(p["L_scen"], p["a"], p["b"], p["gas"], p["F"], alpha=al, lam=2.0)
        costs[al] = res["scenario_costs"]
    tail_lo = empirical_cvar(costs[0.8], 0.99)
    tail_hi = empirical_cvar(costs[0.99], 0.99)
    assert tail_hi <= tail_lo + 1e-3
