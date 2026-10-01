"""In-sample experiments: base case, baselines, required sensitivity sweeps.

Usage:
    PYTHONPATH=src python scripts/03_solve_and_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amm_cvar import config as C
from amm_cvar import plots
from amm_cvar import sensitivity as sens
from amm_cvar.baselines import (
    equal_split,
    evaluate_allocation,
    liquidity_proportional,
    mean_only_allocation,
    single_pool_best_liquidity,
    single_pool_best_mean,
)
from amm_cvar.metrics import empirical_cvar, empirical_var, summarize_losses
from amm_cvar.optimizer import problem_dimensions, solve_cvar_routing
from amm_cvar.pools import build_pool_state
from amm_cvar.scenarios import trailing_scenarios


def main() -> None:
    print("=" * 70)
    print("03 IN-SAMPLE: SOLVE, BASELINES, SENSITIVITY")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load processed real data
    # ------------------------------------------------------------------
    meta = pd.read_csv(C.PROCESSED_DIR / "pools_meta.csv")
    tvl = pd.read_csv(C.PROCESSED_DIR / "tvl_daily.csv", index_col=0, parse_dates=True)
    tvl.index = pd.to_datetime(tvl.index, utc=True)
    eth_price = json.loads((C.PROCESSED_DIR / "eth_price.json").read_text())["eth_price"]

    chart_paths = {p.stem: str(p) for p in (C.RAW_DIR / "defillama_charts").glob("*.json")}
    from amm_cvar.pools import pool_registry_from_snapshot

    meta_full, charts = pool_registry_from_snapshot(
        C.RAW_DIR / "defillama_pools.json", chart_paths,
        gas_gwei=float((C.PROCESSED_DIR / "gas_gwei.txt").read_text().strip()),
        eth_price=eth_price,
    )
    # restrict to the selected pools (order = tvl columns)
    meta = meta.set_index("name").loc[list(tvl.columns)].reset_index()
    state = build_pool_state(meta, {r["pool_id"]: charts[r["pool_id"]] for _, r in meta.iterrows()}, eth_price)
    a, b, gas, names = state.a, state.b, state.gas_usd, state.names
    M = state.M

    # reference decision date: middle of the trade window
    ref_date = pd.Timestamp(C.TRADE_START, tz="UTC") + pd.Timedelta(days=15)
    scen = trailing_scenarios(tvl, ref_date, window=C.SCENARIO_WINDOW)
    K = scen["L_scenarios"].shape[1]
    L_scen = scen["L_scenarios"]
    F = C.BASE_ORDER_USD

    print(f"\nReal pools (M={M}):")
    for i, n in enumerate(names):
        print(f"  [{i}] {n:40s} phi={a[i]*100:.3f}%  L_nom=${scen['L_nominal'][i]:,.0f}  g=${gas[i]:.2f}")
    print(f"\nScenarios: K={K}  (trailing {C.SCENARIO_WINDOW} real TVL-drawdown days, ref={ref_date.date()})")
    d = scen["deltas"]
    print(f"  delta stats: mean={d.mean():.3f}  p50={np.median(d):.3f}  p95={np.quantile(d, 0.95):.3f}  max={d.max():.3f}")
    print(f"Order size F = ${F:,.0f}   alpha={C.ALPHA}  lambda={C.LAMBDA}  eta={C.ETA}")

    dims = problem_dimensions(M, K)
    print(f"Problem size: {dims['n_variables']} variables, {dims['n_constraints']} constraints "
          f"(f: {M}, z: {M}, t: 1, u: {K})")

    # warmup solve so reported timings exclude CVXPY compilation
    solve_cvar_routing(L_scen, a, b, gas, F, alpha=C.ALPHA, lam=C.LAMBDA, eta=C.ETA)

    # ------------------------------------------------------------------
    # Base case
    # ------------------------------------------------------------------
    res = solve_cvar_routing(L_scen, a, b, gas, F, alpha=C.ALPHA, lam=C.LAMBDA, eta=C.ETA)
    print("\n" + "=" * 70)
    print("BASE CASE  (CVaR router)")
    print("=" * 70)
    print(f"  solver={res['solver']}  status={res['status']}  solve_time={res['solve_time']*1e3:.2f} ms")
    print(f"  optimal objective = {res['optimal_value']:.2f}")
    print(f"  mean scenario cost = {res['mean_cost']:.2f}")
    print(f"  CVaR_{C.ALPHA} = {res['cvar_value']:.2f}   (t* = {res['cvar_t']:.2f})")
    alloc_df = pd.DataFrame({
        "pool": names, "project": state.projects,
        "fee": a, "L_nominal": scen["L_nominal"], "gas_usd": gas,
        "f_m": res["allocation"], "share": res["allocation"] / F,
        "z_m": res["usage"],
    })
    print(alloc_df.to_string(index=False))
    alloc_df.to_csv(C.TABLES_DIR / "base_case_allocation.csv", index=False)
    (C.TABLES_DIR / "base_case_meta.json").write_text(json.dumps({
        "M": M, "K": K, "F": F, "alpha": C.ALPHA, "lam": C.LAMBDA, "eta": C.ETA,
        "ref_date": str(ref_date.date()), "solver": res["solver"],
        "solve_time_ms": res["solve_time"] * 1e3,
        "optimal_value": res["optimal_value"],
        "mean_cost": res["mean_cost"], "cvar_value": res["cvar_value"],
        **dims,
    }, indent=2))

    # ------------------------------------------------------------------
    # Baselines (project deliverable)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)
    strategies = {
        "CVaR (ours)": res["allocation"],
        "MeanOnly (lambda=0)": mean_only_allocation(F, L_scen, a, b, gas)["allocation"],
        "SinglePoolBest (mean)": single_pool_best_mean(F, L_scen, a, b, gas)["allocation"],
        "SinglePoolDeep (max L)": single_pool_best_liquidity(F, L_scen, a, b, gas)["allocation"],
        "Equal split (extra)": equal_split(F, L_scen, a, b, gas)["allocation"],
        "Liq-proportional (extra)": liquidity_proportional(F, L_scen, a, b, gas)["allocation"],
    }
    rows, scen_costs = [], {}
    for label, f in strategies.items():
        ev = evaluate_allocation(f, L_scen, a, b, gas, alpha=C.ALPHA)
        rows.append({
            "strategy": label,
            "mean_cost": ev["mean_cost"],
            f"VaR_{C.ALPHA}": ev["var_value"],
            f"CVaR_{C.ALPHA}": ev["cvar_value"],
            "max_cost": ev["max_cost"],
            "n_pools": int((f > C.ACTIVE_POOL_TOL_FRAC * F).sum()),
            "allocation": ",".join(f"{x:.2f}" for x in f),
        })
        scen_costs[label] = ev["scenario_costs"]
    base_df = pd.DataFrame(rows).set_index("strategy")
    print(base_df.drop(columns=["allocation"]).to_string())
    base_df.to_csv(C.TABLES_DIR / "baselines_insample.csv")
    pd.to_pickle(strategies, C.TABLES_DIR / "baseline_allocations.pkl")

    cv = base_df.loc["CVaR (ours)", f"CVaR_{C.ALPHA}"]
    for label in ("SinglePoolBest (mean)", "MeanOnly (lambda=0)"):
        other = base_df.loc[label, f"CVaR_{C.ALPHA}"]
        print(f"  CVaR reduction vs {label}: {100*(other-cv)/other:.2f}%")

    plots.fig_base_case(res["allocation"], names, F, scen_costs, C.ALPHA)
    plots.fig_baselines(base_df.drop(columns=["allocation"]), scen_costs, C.ALPHA)

    # ------------------------------------------------------------------
    # Sensitivity 1: alpha
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY 1: alpha (risk aversion)")
    print("=" * 70)
    df_alpha = sens.sweep_alpha(scen, a, b, gas, F)
    show = df_alpha.drop(columns=["allocation"]).copy()
    for col in ("mean_cost", "CVaR_eval", "CVaR_own_alpha", "VaR_eval", "max_cost"):
        show[col] = show[col].round(2)
    print(show.to_string(index=False))
    df_alpha.drop(columns=["allocation"]).to_csv(C.TABLES_DIR / "sensitivity_alpha.csv", index=False)
    pd.to_pickle(df_alpha, C.TABLES_DIR / "sensitivity_alpha.pkl")
    plots.fig_sensitivity_alpha(df_alpha, names, F)

    # ------------------------------------------------------------------
    # Sensitivity 2: order size
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY 2: order size F (when to split)")
    print("=" * 70)
    df_size = sens.sweep_order_size(scen, a, b, gas)
    show = df_size.drop(columns=["allocation", "allocation_frac"]).copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].round(3)
    print(show.to_string(index=False))
    df_size.drop(columns=["allocation", "allocation_frac"]).to_csv(C.TABLES_DIR / "sensitivity_size.csv", index=False)
    pd.to_pickle(df_size, C.TABLES_DIR / "sensitivity_size.pkl")
    plots.fig_sensitivity_size(df_size, names)

    # ------------------------------------------------------------------
    # Sensitivity 3: uncertainty intensity
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY 3: uncertainty intensity delta")
    print("=" * 70)
    df_delta = sens.sweep_uncertainty(tvl, a, b, gas, F, ref_date)
    show = df_delta.drop(columns=["allocation"]).copy()
    for col in ("mean_cost", "CVaR", "VaR", "max_cost"):
        show[col] = show[col].round(2)
    print(show.to_string(index=False))
    df_delta.drop(columns=["allocation"]).to_csv(C.TABLES_DIR / "sensitivity_delta.csv", index=False)
    pd.to_pickle(df_delta, C.TABLES_DIR / "sensitivity_delta.pkl")
    plots.fig_sensitivity_delta(df_delta.drop(columns=["allocation"]))

    # ------------------------------------------------------------------
    # Extras: frontier + scalability
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRA: efficient frontier (lambda)")
    print("=" * 70)
    df_front = sens.sweep_lambda(scen, a, b, gas, F)
    print(df_front.drop(columns=["allocation"]).round(2).to_string(index=False))
    df_front.drop(columns=["allocation"]).to_csv(C.TABLES_DIR / "frontier.csv", index=False)
    pd.to_pickle(df_front, C.TABLES_DIR / "frontier.pkl")
    plots.fig_frontier(df_front)

    print("\n" + "=" * 70)
    print("EXTRA: scalability (M, K)")
    print("=" * 70)
    df_scale = sens.sweep_scalability()
    print(df_scale.round(2).to_string(index=False))
    df_scale.to_csv(C.TABLES_DIR / "scalability.csv", index=False)
    plots.fig_scalability(df_scale)

    # data overview figure
    deltas_hist = np.load(C.PROCESSED_DIR / "delta_hist.npy")
    orders = pd.read_csv(C.PROCESSED_DIR / "parent_orders.csv")
    plots.fig_data_overview(tvl, orders, deltas_hist.ravel(), names)

    print("\nDone. Tables in results/tables, figures in results/figures.")


if __name__ == "__main__":
    main()
