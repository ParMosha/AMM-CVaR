"""Baseline routing strategies (project deliverable 5 + extras).

Mandatory baselines
-------------------
* ``single_pool_best_mean``   -- entire order in the pool with lowest mean cost
* ``mean_only_allocation``    -- same problem with lambda = 0 (scenario mean only)

Extras (labelled as such in the report)
---------------------------------------
* ``single_pool_best_liquidity`` -- entire order in the deepest pool
* ``equal_split``
* ``liquidity_proportional``
"""

from __future__ import annotations

import numpy as np

from .cost_model import scenario_cost_numpy, true_execution_cost
from .optimizer import solve_cvar_routing


def _single_pool_costs(F, L_scen, a, b, gas) -> np.ndarray:
    """Mean *true* scenario cost of sending everything to each pool m."""
    M = L_scen.shape[0]
    out = np.zeros(M)
    for m in range(M):
        f = np.zeros(M)
        f[m] = F
        out[m] = np.mean([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ])
    return out


def single_pool_best_mean(F, L_scen, a, b, gas) -> dict:
    costs = _single_pool_costs(F, L_scen, a, b, gas)
    m = int(np.argmin(costs))
    f = np.zeros(L_scen.shape[0])
    f[m] = F
    z = np.zeros_like(f)
    z[m] = 1.0
    return {
        "name": "single-pool best (mean cost)",
        "allocation": f,
        "usage": z,
        "pool": m,
        "mean_cost": float(costs[m]),
        "scenario_costs": np.array([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ]),
    }


def single_pool_best_liquidity(F, L_scen, a, b, gas) -> dict:
    m = int(np.argmax(L_scen.mean(axis=1)))
    f = np.zeros(L_scen.shape[0])
    f[m] = F
    z = np.zeros_like(f)
    z[m] = 1.0
    return {
        "name": "single-pool deepest (max L)",
        "allocation": f,
        "usage": z,
        "pool": m,
        "mean_cost": float(scenario_cost_numpy(f, L_scen, a, b, gas=gas, z=z).mean()),
        "scenario_costs": np.array([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ]),
    }


def equal_split(F, L_scen, a, b, gas) -> dict:
    M = L_scen.shape[0]
    f = np.full(M, F / M)
    z = np.ones(M)
    return {
        "name": "equal split",
        "allocation": f,
        "usage": z,
        "mean_cost": float(np.mean([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ])),
        "scenario_costs": np.array([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ]),
    }


def liquidity_proportional(F, L_scen, a, b, gas) -> dict:
    L = L_scen.mean(axis=1)
    f = F * L / L.sum()
    z = np.ones_like(f)
    return {
        "name": "liquidity proportional",
        "allocation": f,
        "usage": z,
        "mean_cost": float(np.mean([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ])),
        "scenario_costs": np.array([
            true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(L_scen.shape[1])
        ]),
    }


def mean_only_allocation(F, L_scen, a, b, gas, **kwargs) -> dict:
    """Solve the *same* problem with lambda = 0 (project baseline)."""
    res = solve_cvar_routing(L_scen, a, b, gas, F, lam=0.0, **kwargs)
    res = dict(res)
    res["name"] = "mean-only (lambda=0)"
    return res


def evaluate_allocation(f, L_scen, a, b, gas, alpha=0.95) -> dict:
    """Scenario-set statistics of a fixed allocation under **true** gas costs.

    Every strategy (including the CVaR router) is evaluated with the economic
    cost actually paid: fees + impact + *fixed* gas per pool touched.  The
    optimiser itself uses the assignment relaxation ``g_m z_m`` with
    ``z_m = f_m / F`` (see README "collapse lemma"); the difference is the
    relaxation gap, largest for small orders.
    """
    from .metrics import summarize_losses

    f = np.asarray(f, dtype=float)
    used = f > 0.0
    z = used.astype(float)
    M, K = L_scen.shape
    costs = np.array([
        true_execution_cost(f, L_scen[:, k], a, b, gas) for k in range(K)
    ])
    stats = summarize_losses(costs, alphas=(alpha,))
    return {
        "allocation": f,
        "usage": z,
        "scenario_costs": costs,
        "mean_cost": stats["mean"],
        "cvar_value": stats[f"CVaR_{alpha:.2f}"],
        "var_value": stats[f"VaR_{alpha:.2f}"],
        "max_cost": stats["max"],
        **stats,
    }
