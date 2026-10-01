"""Sensitivity studies required by the project statement.

(1) sweep over alpha            -- risk aversion
(2) sweep over order size F     -- when does splitting become necessary
(3) sweep over uncertainty delta-- how much does CVaR protect the plan

Extras: efficient frontier in lambda, scalability in (M, K).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .baselines import evaluate_allocation, mean_only_allocation, single_pool_best_mean
from .optimizer import problem_dimensions, solve_cvar_routing
from .scenarios import scenario_cost_matrix, trailing_scenarios


def sweep_alpha(scen, a, b, gas, F, alphas=None, lam=None, eta=None, alpha_eval=0.95):
    alphas = alphas or C.ALPHA_GRID
    lam = C.LAMBDA if lam is None else lam
    eta = C.ETA if eta is None else eta
    L_scen = scen["L_scenarios"]
    rows = []
    for al in alphas:
        res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=al, lam=lam, eta=eta)
        ev = evaluate_allocation(res["allocation"], L_scen, a, b, gas, alpha=alpha_eval)
        ev_self = evaluate_allocation(res["allocation"], L_scen, a, b, gas, alpha=al)
        rows.append({
            "alpha": al,
            "mean_cost": ev["mean_cost"],
            "CVaR_eval": ev["cvar_value"],
            "CVaR_own_alpha": ev_self["cvar_value"],
            "VaR_eval": ev["var_value"],
            "max_cost": ev["max_cost"],
            "solve_time_ms": res["solve_time"] * 1e3,
            "allocation": res["allocation"].copy(),
            "n_active": int((res["allocation"] > C.ACTIVE_POOL_TOL_FRAC * F).sum()),
        })
    return pd.DataFrame(rows)


def sweep_order_size(scen, a, b, gas, F_grid=None, alpha=None, lam=None, eta=None):
    F_grid = F_grid or C.F_GRID_USD
    alpha = C.ALPHA if alpha is None else alpha
    lam = C.LAMBDA if lam is None else lam
    eta = C.ETA if eta is None else eta
    L_scen = scen["L_scenarios"]
    rows = []
    for F in F_grid:
        res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
        sp = single_pool_best_mean(F, L_scen, a, b, gas)
        ev = evaluate_allocation(res["allocation"], L_scen, a, b, gas, alpha=alpha)
        sp_costs = sp["scenario_costs"]
        rows.append({
            "F": F,
            "n_active": int((res["allocation"] > C.ACTIVE_POOL_TOL_FRAC * F).sum()),
            "mean_cost": ev["mean_cost"],
            "mean_cost_per_usd_bps": 1e4 * ev["mean_cost"] / F,
            "CVaR": ev["cvar_value"],
            "CVaR_per_usd_bps": 1e4 * ev["cvar_value"] / F,
            "max_cost": ev["max_cost"],
            "single_pool_mean": float(np.mean(sp_costs)),
            "solve_time_ms": res["solve_time"] * 1e3,
            "allocation": res["allocation"].copy(),
            "allocation_frac": res["allocation"] / F,
        })
    return pd.DataFrame(rows)


def sweep_uncertainty(
    tvl,
    a,
    b,
    gas,
    F,
    date,
    scales=None,
    alpha=None,
    lam=None,
    eta=None,
    window=None,
    delta_bar=None,
):
    """How conservative does the CVaR plan become as delta intensity grows?"""
    scales = scales or C.DELTA_SCALE_GRID
    alpha = C.ALPHA if alpha is None else alpha
    lam = C.LAMBDA if lam is None else lam
    eta = C.ETA if eta is None else eta
    window = C.SCENARIO_WINDOW if window is None else window
    delta_bar = C.DELTA_BAR if delta_bar is None else delta_bar

    rows = []
    for scale in scales:
        scen = trailing_scenarios(tvl, date, window=window, delta_bar=delta_bar, delta_scale=scale)
        L_scen = scen["L_scenarios"]
        cvar = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
        mean_only = mean_only_allocation(F, L_scen, a, b, gas, alpha=alpha, eta=eta)
        for tag, res in (("CVaR", cvar), ("MeanOnly", mean_only)):
            ev = evaluate_allocation(res["allocation"], L_scen, a, b, gas, alpha=alpha)
            rows.append({
                "delta_scale": scale,
                "strategy": tag,
                "mean_cost": ev["mean_cost"],
                "CVaR": ev["cvar_value"],
                "VaR": ev["var_value"],
                "max_cost": ev["max_cost"],
                "n_active": int((res["allocation"] > C.ACTIVE_POOL_TOL_FRAC * F).sum()),
                "allocation": res["allocation"].copy(),
            })
    return pd.DataFrame(rows)


def sweep_lambda(scen, a, b, gas, F, lam_grid=None, alpha=None, eta=None):
    lam_grid = C.LAMBDA_GRID if lam_grid is None else lam_grid
    alpha = C.ALPHA if alpha is None else alpha
    eta = C.ETA if eta is None else eta
    L_scen = scen["L_scenarios"]
    rows = []
    for lam in lam_grid:
        res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
        ev = evaluate_allocation(res["allocation"], L_scen, a, b, gas, alpha=alpha)
        rows.append({
            "lam": lam,
            "mean_cost": ev["mean_cost"],
            "CVaR": ev["cvar_value"],
            "max_cost": ev["max_cost"],
            "allocation": res["allocation"].copy(),
        })
    return pd.DataFrame(rows)


def sweep_scalability(a_ranges=None, seed: int = C.SEED, F=None, alpha=None, lam=None, eta=None):
    """Solve time vs (M, K) using random pools in the real L / fee ranges.

    Reports cold time (includes CVXPY compilation of the DPP problem) and
    warm time (parameter update + re-solve) separately.
    """
    rng = C.set_seed(seed)
    alpha = C.ALPHA if alpha is None else alpha
    lam = C.LAMBDA if lam is None else lam
    eta = C.ETA if eta is None else eta
    F = C.BASE_ORDER_USD if F is None else F
    rows = []
    for M, K in C.SCALABILITY_GRID:
        L_nom = rng.uniform(2e6, 8e7, size=M)          # TVL/2 in realistic USD range
        a = rng.uniform(0.0001, 0.003, size=M)
        b = np.ones(M)
        gas = rng.uniform(2.0, 15.0, size=M)
        xi = np.sort(rng.beta(2, 5, size=K))
        deltas = np.outer(rng.uniform(0.05, 0.45, size=M), xi)
        L_scen = L_nom[:, None] * (1.0 - np.clip(deltas, 0, 0.6))
        t0 = time.perf_counter()
        res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(5):
            res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
        warm = (time.perf_counter() - t0) / 5
        dims = problem_dimensions(M, K)
        rows.append({
            "M": M, "K": K,
            "cold_time_ms": 1e3 * cold,
            "solve_time_ms": 1e3 * warm,
            "solver": res["solver"],
            "status": res["status"],
            **dims,
        })
    return pd.DataFrame(rows)
