"""CVXPY implementation of the project optimisation problem (exact statement).

    min_{f,z,t,u}   (1/K) sum_k C_k(f)  +  lambda * CVaR_alpha(C)  +  eta * sum_m z_m

    s.t.            sum_m f_m = F,   f_m >= 0
                    0 <= z_m <= 1,   f_m <= F z_m
                    u_k >= C_k(f) - t,  u_k >= 0

    C_k(f) = sum_m ( a_m f_m + b_m f_m^2 / L_{m,k} )  +  sum_m g_m z_m

    CVaR_alpha(C) = min_{t,u}  t + 1/((1-alpha)K) sum_k u_k

Convexity / DCP
---------------
* ``sum_m b_m f_m^2 / L_{m,k}`` is DCP-equal to the assignment hint
  ``sum_m b_m quad_over_lin(f_m, L_{m,k})`` and is written in the vectorised
  form  ``|| diag(sqrt(b/L_k)) f ||_2^2``  (identical cone, less overhead).
* The CVaR term is the Rockafellar--Uryasev epigraph (affine in ``(t,u)``).

Numerics
--------
Internally the problem is solved in *allocation shares* ``x = f / F`` with all
costs divided by ``F``.  The argmin is unchanged (the USD objective is exactly
``F`` times the scaled objective) and the problem is DPP-compliant, so a single
CVXPY problem is compiled per ``(M, K, alpha, lam)`` and re-solved for every
order size / liquidity scenario through parameters.  This matters for the
2,500-order walk-forward backtest.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import cvxpy as cp
import numpy as np

from . import config as C


def problem_dimensions(M: int, K: int) -> dict[str, int]:
    """Variable / constraint counts of the formulation (for the report)."""
    return {
        "M": M,
        "K": K,
        "n_variables": M + M + 1 + K,          # f, z, t, u
        "n_constraints": 1 + M + M + K + K,    # budget, z<=1, f<=Fz, u>=C-t, u>=0
    }


# ---------------------------------------------------------------------------
# Cached, parameterised (DPP) problem
# ---------------------------------------------------------------------------
class _RoutingProblem:
    """One compiled problem, re-solved many times via parameters."""

    def __init__(self, M: int, K: int, alpha: float, lam: float):
        self.M, self.K, self.alpha, self.lam = M, K, alpha, lam

        self.x = cp.Variable(M, nonneg=True, name="x")
        self.z = cp.Variable(M, nonneg=True, name="z")
        self.p_slip = cp.Parameter((M, K), nonneg=True, name="p_slip")   # sqrt(F * b / L)
        self.p_a = cp.Parameter(M, name="p_a")                           # fee coefficients
        self.p_g = cp.Parameter(M, nonneg=True, name="p_g")              # gas / F
        self.p_eta = cp.Parameter(nonneg=True, name="p_eta")             # eta / F

        slip = [cp.sum_squares(cp.multiply(self.p_slip[:, k], self.x)) for k in range(K)]
        self.C_k = [self.p_a @ self.x + slip[k] + self.p_g @ self.z for k in range(K)]
        C_vec = cp.hstack(self.C_k)
        mean_expr = (1.0 / K) * cp.sum(C_vec)

        cons = [cp.sum(self.x) == 1.0, self.z <= 1.0, self.x <= self.z]
        if lam > 0:
            self.t = cp.Variable(name="t")
            self.u = cp.Variable(K, nonneg=True, name="u")
            cons.append(self.u >= C_vec - self.t)
            cvar_expr = self.t + (1.0 / ((1.0 - alpha) * K)) * cp.sum(self.u)
            obj = mean_expr + lam * cvar_expr + self.p_eta * cp.sum(self.z)
        else:
            self.t = None
            self.u = None
            obj = mean_expr + self.p_eta * cp.sum(self.z)

        self.problem = cp.Problem(cp.Minimize(obj), cons)
        if not self.problem.is_dcp():
            raise RuntimeError("Routing problem failed DCP check")
        self._solved_ok = False

    def solve(self, p_slip, p_a, p_g, p_eta, solver: str):
        self.p_slip.value = p_slip
        self.p_a.value = p_a
        self.p_g.value = p_g
        self.p_eta.value = p_eta
        self.problem.solve(solver=solver, verbose=False)
        return self.problem.status


_CACHE: dict[tuple, _RoutingProblem] = {}


def _get_problem(M: int, K: int, alpha: float, lam: float) -> _RoutingProblem:
    key = (M, K, round(float(alpha), 6), round(float(lam), 6))
    if key not in _CACHE:
        _CACHE[key] = _RoutingProblem(M, K, alpha, lam)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_cvar_routing(
    L_scenarios: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gas: np.ndarray,
    F: float,
    alpha: float = C.ALPHA,
    lam: float = C.LAMBDA,
    eta: float = C.ETA,
    solver: str | None = None,
    warm_start_vars: dict[str, Any] | None = None,
) -> dict:
    """Solve the CVaR order-routing problem.

    Parameters
    ----------
    L_scenarios : (M, K) liquidity index per pool and scenario (USD)
    a, b : (M,) cost coefficients  (c_m = a_m f_m + b_m f_m^2 / L_m)
    gas : (M,) fixed gas cost per pool usage (USD)
    F : order size (USD)
    alpha : CVaR confidence level in (0, 1)
    lam : weight of CVaR in the objective (lam = 0 -> mean-only baseline)
    eta : pool-usage penalty (provably constant at optimum, kept for fidelity)

    Returns
    -------
    dict with allocation ``f`` (USD), usage ``z``, CVaR / mean costs (USD),
    solve time and problem dimensions.
    """
    L_scen = np.asarray(L_scenarios, dtype=float)
    if L_scen.ndim != 2:
        raise ValueError("L_scenarios must be (M, K)")
    M, K = L_scen.shape
    a = np.asarray(a, dtype=float).reshape(M)
    b = np.asarray(b, dtype=float).reshape(M)
    gas = np.asarray(gas, dtype=float).reshape(M)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if F <= 0:
        raise ValueError("F must be positive")
    if np.any(L_scen <= 0):
        raise ValueError("L_scenarios must be strictly positive")
    if np.any(b < 0) or np.any(a < 0):
        raise ValueError("cost coefficients must be non-negative")

    # scaled parameters (x = f / F, costs / F)
    p_slip = np.sqrt(F * b[:, None] / L_scen)     # (M, K)
    p_a = a.copy()
    p_g = gas / F
    p_eta = float(eta) / F

    prob = _get_problem(M, K, alpha, lam)

    solvers_try = [solver] if solver else [C.SOLVER_PRIMARY, C.SOLVER_FALLBACK]
    last_err: Exception | None = None
    start = time.perf_counter()
    status = None
    used_solver = None
    for s in solvers_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                status = prob.solve(p_slip, p_a, p_g, p_eta, solver=s)
            used_solver = s
            if status in ("optimal", "optimal_inaccurate") and prob.x.value is not None:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            status = f"error: {exc}"
            continue
    solve_time = time.perf_counter() - start

    if status not in ("optimal", "optimal_inaccurate") or prob.x.value is None:
        raise RuntimeError(f"Solve failed (status={status}, err={last_err})")

    x_val = np.maximum(np.asarray(prob.x.value, dtype=float), 0.0)
    x_val = x_val / x_val.sum()                   # enforce budget exactly
    f_val = x_val * F
    # numerical dust -> exact zeros.  Anything below 1bp of the order is
    # economically irrelevant but would otherwise make the "pool used" test of
    # the true fixed-gas cost fire for every pool the solver touched.
    dust = 1e-4 * F
    f_val[f_val < dust] = 0.0
    if f_val.sum() > 0:
        f_val = f_val * (F / f_val.sum())
    z_val = np.asarray(prob.z.value, dtype=float)
    z_val = np.maximum(z_val, f_val / F)          # feasibility of f <= F z

    # evaluation stats in USD on the scenario set, under the *true* fixed-gas
    # cost (the same criterion used for every strategy / the backtest).  The
    # optimiser itself minimises the relaxed objective of the assignment.
    from .cost_model import true_execution_cost

    costs = np.array([true_execution_cost(f_val, L_scen[:, k], a, b, gas) for k in range(K)])
    mean_cost = float(costs.mean())
    from .metrics import empirical_cvar

    cvar_value = empirical_cvar(costs, alpha)
    t_val = float(prob.t.value * F) if prob.t is not None else None

    return {
        "status": status,
        "solver": used_solver,
        "solve_time": solve_time,
        "optimal_value": float(prob.problem.value * F),
        "allocation": f_val,
        "usage": z_val,
        "scenario_costs": costs,
        "mean_cost": mean_cost,
        "cvar_value": cvar_value,
        "cvar_t": t_val,
        "alpha": alpha,
        "lam": lam,
        "eta": eta,
        "F": F,
        "dimensions": problem_dimensions(M, K),
    }


def _eval_ck(f, z, a, b, gas, L_k) -> float:
    return float(a @ f + np.sum(b * f**2 / L_k) + gas @ z)


def solve_cvar_routing_quadratic_over_linear_reference(
    L_scenarios, a, b, gas, F, alpha=0.95, lam=1.0, eta=0.0,
) -> dict:
    """Literal ``quad_over_lin`` build of the assignment (slow reference).

    Kept for documentation / numerical cross-checks against the vectorised
    formulation used in :func:`solve_cvar_routing`.
    """
    L_scen = np.asarray(L_scenarios, dtype=float)
    M, K = L_scen.shape
    a = np.asarray(a, float).reshape(M)
    b = np.asarray(b, float).reshape(M)
    gas = np.asarray(gas, float).reshape(M)

    f = cp.Variable(M, nonneg=True)
    z = cp.Variable(M, nonneg=True)
    t = cp.Variable()
    u = cp.Variable(K, nonneg=True)
    C_k = []
    for k in range(K):
        slip = cp.sum(cp.hstack([b[m] * cp.quad_over_lin(f[m], L_scen[m, k]) for m in range(M)]))
        C_k.append(a @ f + slip + gas @ z)
    C_vec = cp.hstack(C_k)
    obj = (1 / K) * cp.sum(C_vec) + lam * (t + (1 / ((1 - alpha) * K)) * cp.sum(u)) + eta * cp.sum(z)
    cons = [cp.sum(f) == F, z <= 1, f <= F * z, u >= C_vec - t]
    prob = cp.Problem(cp.Minimize(obj), cons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            prob.solve(solver=C.SOLVER_PRIMARY)
        except Exception:  # noqa: BLE001
            prob.solve(solver=C.SOLVER_FALLBACK)
    f_val = np.maximum(np.asarray(f.value, float), 0.0)
    z_val = np.asarray(z.value, float)
    costs = np.array([_eval_ck(f_val, z_val, a, b, gas, L_scen[:, k]) for k in range(K)])
    from .metrics import empirical_cvar

    return {
        "allocation": f_val,
        "usage": z_val,
        "scenario_costs": costs,
        "mean_cost": float(costs.mean()),
        "cvar_value": empirical_cvar(costs, alpha),
        "status": prob.status,
    }
