"""Convex AMM execution-cost model.

Model (per the project statement, USD notionals)::

    c_m(f_m; L_m)  =  a_m f_m  +  b_m f_m^2 / L_m

where ``a_m = phi_m`` is the pool fee rate and ``b_m f^2/L`` is the
second-order price impact of a constant-product pool with liquidity index
``L``.  See ``README.md`` for the derivation from ``xy = k``.

This module provides
  * the exact ``xy=k`` impact (used to justify the quadratic surrogate),
  * vectorised evaluation of scenario costs (for baselines / evaluation),
  * the *true* execution cost (fixed gas per pool actually touched),
  * a least-squares impact-shape calibration on the real Binance trade tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cost primitives
# ---------------------------------------------------------------------------
def xyk_exact_impact(n: np.ndarray | float, L: np.ndarray | float) -> np.ndarray:
    """Exact slippage cost (USD) of selling notional ``n`` on ``xy=k``.

    Selling ``n`` USD of X against reserves with output-side reserve ``y = L``
    costs ``n^2 / (L + n)`` USD of price impact (derived in README).
    """
    n = np.asarray(n, dtype=float)
    L = np.asarray(L, dtype=float)
    return n**2 / (L + n)


def quadratic_impact(n: np.ndarray | float, L: np.ndarray | float, b: float | np.ndarray = 1.0) -> np.ndarray:
    """Convex surrogate ``b n^2 / L`` (second-order Taylor of :func:`xyk_exact_impact`)."""
    n = np.asarray(n, dtype=float)
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    return b * n**2 / L


def scenario_cost_numpy(
    f: np.ndarray,
    L_scen: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gas: np.ndarray | None = None,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Scenario costs ``C_k(f)`` for allocation ``f``.

    Parameters
    ----------
    f : (M,) allocation in USD
    L_scen : (M, K) liquidity per pool and scenario
    a, b : (M,) cost coefficients
    gas : (M,) fixed gas per pool (paid via the usage variable ``z``)
    z : (M,) usage variables; defaults to ``f / sum(f)`` (the optimum of the
        relaxation, see README "collapse lemma")

    Returns
    -------
    (K,) array of total costs in USD.
    """
    f = np.asarray(f, dtype=float)
    L_scen = np.atleast_2d(np.asarray(L_scen, dtype=float))
    if L_scen.shape[0] != f.shape[0]:
        L_scen = L_scen.T
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    K = L_scen.shape[1]

    fee = a @ f                                   # (scalar)
    impact = np.empty(K)
    for k in range(K):
        impact[k] = np.sum(b * f**2 / L_scen[:, k])
    out = fee + impact
    if gas is not None:
        g = np.asarray(gas, dtype=float)
        if z is None:
            z = f / max(f.sum(), 1e-12)
        out = out + float(g @ z)
    return out


def true_execution_cost(
    f: np.ndarray,
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gas: np.ndarray,
    used_tol: float = 0.0,
) -> float:
    """Economic cost actually paid: fees + impact + *fixed* gas per used pool.

    Unlike the relaxed optimisation model (which spreads gas through
    ``g_m z_m`` with ``z_m = f_m / F``), realised gas is a true fixed cost
    per pool touched.  All strategies are evaluated with this criterion.
    """
    f = np.asarray(f, dtype=float)
    L = np.asarray(L, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gas = np.asarray(gas, dtype=float)
    used = f > used_tol
    fee = float(a @ f)
    impact = float(np.sum(b[used] * f[used] ** 2 / L[used]))
    g = float(gas[used].sum())
    return fee + impact + g


# ---------------------------------------------------------------------------
# Impact-shape calibration on the real trade tape
# ---------------------------------------------------------------------------
def fit_impact_regression(impact_df: pd.DataFrame, min_notional: float = 2_000.0, n_bins: int = 18) -> dict:
    """Least-squares fit  cost ~= a*n + b*n^2  on real realised impacts.

    Individual 1-second observations are dominated by bid/ask bounce, so the
    fit is performed on **log-size bin means** (averaging kills the bounce and
    exposes the impact shape).  This is a *model-fidelity check* on the Binance
    tape (a CEX limit-order book) used to support the fee + quadratic-impact
    structure; it is **not** used to calibrate the AMM pools themselves (those
    use real TVL + protocol fees).
    """
    df = impact_df if isinstance(impact_df, pd.DataFrame) else pd.DataFrame(impact_df)
    df = df.dropna().copy()
    df = df[(df["notional"] > 0) & np.isfinite(df["cost_usd"])]
    df = df[df["notional"] >= min_notional]
    df = df[df["cost_usd"] > -0.05 * df["notional"]]
    if len(df) < 50:
        return {"a_tape": float("nan"), "b_tape": float("nan"), "r2": float("nan"),
                "n_obs": int(len(df)), "median_notional": float("nan"), "p95_notional": float("nan")}

    edges = np.unique(np.quantile(df["notional"], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 4:
        edges = np.unique(np.quantile(df["notional"], np.linspace(0, 1, 9)))
    df["bin"] = pd.cut(df["notional"], bins=edges, include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("notional", "size"),
        mean_notional=("notional", "mean"),
        mean_cost=("cost_usd", "mean"),
        se_cost=("cost_usd", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
    ).reset_index(drop=True)
    grouped = grouped[grouped["n"] >= 20].reset_index(drop=True)

    n = grouped["mean_notional"].to_numpy(float)
    y = grouped["mean_cost"].to_numpy(float)
    w = grouped["n"].to_numpy(float)
    X = np.column_stack([n, n**2])
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # descriptive shape checks on the bin means
    cost_per_usd = y / n
    mono = float(np.corrcoef(np.argsort(np.argsort(n)), np.argsort(np.argsort(y)))[0, 1])
    return {
        "a_tape": float(coef[0]),
        "b_tape": float(coef[1]),
        "r2": r2,
        "n_obs": int(len(df)),
        "n_bins": int(len(grouped)),
        "median_notional": float(np.median(df["notional"])),
        "p95_notional": float(np.quantile(df["notional"], 0.95)),
        "cost_increasing": bool(y[-1] > y[0]),
        "spearman_size_cost": mono,
        "bps_at_smallest_bin": float(1e4 * cost_per_usd[0]),
        "bps_at_largest_bin": float(1e4 * cost_per_usd[-1]),
        "bin_means": grouped.to_dict(orient="list"),
    }
