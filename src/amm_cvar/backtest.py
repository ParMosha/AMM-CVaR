"""Walk-forward backtest of the CVaR router on *real* trades.

Design (no look-ahead)
----------------------
For every real parent order with timestamp ``t`` (taker-sell notional on
Binance ETHUSDT aggregated to ``PARENT_FREQ``):

1. ``L_nominal`` and the scenario set use only TVL history **strictly before**
   day(t)  -> the information set of the router.
2. The routing problem is solved for the CVaR strategy and for every baseline.
3. Realised cost is evaluated with the *actual* pool state of day(t)
   (out-of-sample) and the **true** fixed gas per pool touched.

Realised-cost metric (identical for every strategy, see cost_model):
    fees + xy=k impact at realised L + sum_m g_m * 1{f_m > tol}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .baselines import (
    equal_split,
    liquidity_proportional,
    mean_only_allocation,
    single_pool_best_liquidity,
    single_pool_best_mean,
)
from .cost_model import true_execution_cost
from .metrics import paired_comparison, summarize_losses
from .optimizer import solve_cvar_routing
from .scenarios import realized_state, trailing_scenarios


def _strategy_allocations(F, scen, a, b, gas, alpha, lam, eta):
    """Return {name: allocation} for the strategies under comparison."""
    from .baselines import evaluate_allocation

    L_scen = scen["L_scenarios"]
    out = {}

    cvar = solve_cvar_routing(L_scen, a, b, gas, F, alpha=alpha, lam=lam, eta=eta)
    out["CVaR"] = cvar["allocation"]

    # gas-guarded variant: the assignment relaxation undercharges fixed gas,
    # so for small orders a split can be worse than one pool under true cost.
    # Fall back to the best single pool whenever it is cheaper on the scenario set.
    sp_mean = single_pool_best_mean(F, L_scen, a, b, gas)
    ev_cvar = evaluate_allocation(cvar["allocation"], L_scen, a, b, gas, alpha=alpha)
    ev_sp = evaluate_allocation(sp_mean["allocation"], L_scen, a, b, gas, alpha=alpha)
    out["CVaR+gasGuard"] = cvar["allocation"] if ev_cvar["mean_cost"] <= ev_sp["mean_cost"] else sp_mean["allocation"]

    mean_only = mean_only_allocation(F, L_scen, a, b, gas, alpha=alpha, eta=eta)
    out["MeanOnly"] = mean_only["allocation"]

    out["SinglePoolMean"] = sp_mean["allocation"]

    spL = single_pool_best_liquidity(F, L_scen, a, b, gas)
    out["SinglePoolDeep"] = spL["allocation"]

    eq = equal_split(F, L_scen, a, b, gas)
    out["EqualSplit"] = eq["allocation"]

    lp = liquidity_proportional(F, L_scen, a, b, gas)
    out["LiqProp"] = lp["allocation"]
    return out


def run_backtest(
    orders: pd.DataFrame,
    tvl: pd.DataFrame,
    a: np.ndarray,
    b: np.ndarray,
    gas: np.ndarray,
    alpha: float = C.ALPHA,
    lam: float = C.LAMBDA,
    eta: float = C.ETA,
    window: int = C.SCENARIO_WINDOW,
    delta_bar: float = C.DELTA_BAR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the walk-forward loop.  Returns one row per real order x strategy."""
    records: list[dict] = []
    scen_cache: dict[pd.Timestamp, dict] = {}
    real_cache: dict[pd.Timestamp, dict] = {}

    orders = orders.copy()
    orders["ts"] = pd.to_datetime(orders["ts"], utc=True)
    orders["day"] = orders["ts"].dt.tz_convert("UTC").dt.normalize()
    orders = orders.sort_values("ts").reset_index(drop=True)

    n = len(orders)
    last_pct = -1
    for i, row in orders.iterrows():
        day = row["day"]
        if day not in scen_cache:
            scen_cache[day] = trailing_scenarios(tvl, day, window=window, delta_bar=delta_bar)
            real_cache[day] = realized_state(tvl, day, delta_bar=delta_bar)
        scen = scen_cache[day]
        real = real_cache[day]

        F = float(row["F"])
        try:
            allocs = _strategy_allocations(F, scen, a, b, gas, alpha, lam, eta)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [warn] solve failed on {day.date()} F={F:.0f}: {exc}")
            continue

        L_real = real["L_real"]
        d_real = real["delta_real"]
        for name, f in allocs.items():
            cost = true_execution_cost(f, L_real, a, b, gas)
            fee = float(a @ f)
            used = f > C.GAS_USED_TOL_USD
            records.append({
                "ts": row["ts"],
                "day": day,
                "F": F,
                "strategy": name,
                "realized_cost": cost,
                "cost_bps": 1e4 * cost / F,
                "fee_cost": fee,
                "gas_cost": float(gas[used].sum()),
                "impact_cost": cost - fee - float(gas[used].sum()),
                "n_pools_used": int(used.sum()),
                "delta_real_mean": float(d_real.mean()),
                "delta_real_max": float(d_real.max()),
                "L_real_min_frac": float((L_real / scen["L_nominal"]).min()),
                "alloc": ",".join(f"{x:.4f}" for x in f),
            })

        if verbose and n > 0:
            pct = int(100 * (i + 1) / n)
            if pct // 10 > last_pct // 10:
                last_pct = pct
                print(f"  backtest progress: {pct:3d}%  ({i + 1}/{n} orders)")

    return pd.DataFrame.from_records(records)


def summarize_backtest(results: pd.DataFrame, alpha: float = C.ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate realised costs per strategy + paired tests vs CVaR and MeanOnly."""
    rows = []
    for strat, grp in results.groupby("strategy"):
        stats = summarize_losses(grp["realized_cost"].to_numpy(), alphas=(alpha,))
        bps = summarize_losses(grp["cost_bps"].to_numpy(), alphas=(alpha,), prefix="bps_")
        rows.append({"strategy": strat, **stats, **bps})
    summary = pd.DataFrame(rows).set_index("strategy")

    pivot = results.pivot_table(index=["ts", "F"], columns="strategy", values="realized_cost")
    pairs = []
    for other in pivot.columns:
        if other == "CVaR":
            continue
        comp = paired_comparison(pivot["CVaR"].to_numpy(), pivot[other].to_numpy(), alpha=alpha)
        comp["vs"] = other
        pairs.append(comp)
    paired = pd.DataFrame(pairs).set_index("vs")
    return summary, paired


def backtest_by_size(results: pd.DataFrame, alpha: float = C.ALPHA) -> pd.DataFrame:
    """Realised CVaR / mean cost per order-size bucket (shows where splitting pays)."""
    df = results.copy()
    bins = [0, 5e3, 2.5e4, 1e5, 5e5, np.inf]
    labels = ["<$5k", "$5k-25k", "$25k-100k", "$100k-500k", ">$500k"]
    df["size_bucket"] = pd.cut(df["F"], bins=bins, labels=labels)
    rows = []
    for (bucket, strat), grp in df.groupby(["size_bucket", "strategy"], observed=True):
        stats = summarize_losses(grp["realized_cost"].to_numpy(), alphas=(alpha,))
        rows.append({"size_bucket": bucket, "strategy": strat, **stats})
    return pd.DataFrame(rows)


def bad_day_comparison(results: pd.DataFrame, alpha: float = C.ALPHA, q: float = 0.8) -> pd.DataFrame:
    """Strategy comparison restricted to the worst liquidity days.

    Day severity is the *mean* adverse deviation across pools (the max saturates
    at ``delta_bar`` whenever any single pool is stressed)."""
    day_severity = results.groupby("day")["delta_real_mean"].max()
    thresh = day_severity.quantile(q)
    bad_days = day_severity[day_severity >= thresh].index
    sub = results[results["day"].isin(bad_days)]
    rows = []
    for strat, grp in sub.groupby("strategy"):
        stats = summarize_losses(grp["realized_cost"].to_numpy(), alphas=(alpha,))
        rows.append({"strategy": strat, "n_orders": len(grp), **stats})
    out = pd.DataFrame(rows).set_index("strategy")
    out.attrs["severity_threshold"] = float(thresh)
    out.attrs["n_bad_days"] = int(len(bad_days))
    return out
