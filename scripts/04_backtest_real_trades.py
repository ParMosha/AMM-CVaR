"""Walk-forward backtest of the CVaR router on the real trade tape.

Usage:
    PYTHONPATH=src python scripts/04_backtest_real_trades.py
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
from amm_cvar.backtest import backtest_by_size, bad_day_comparison, run_backtest, summarize_backtest
from amm_cvar.cost_model import fit_impact_regression
from amm_cvar.pools import build_pool_state, pool_registry_from_snapshot


def main() -> None:
    print("=" * 70)
    print("04 BACKTEST ON REAL TRADES")
    print("=" * 70)

    meta = pd.read_csv(C.PROCESSED_DIR / "pools_meta.csv")
    tvl = pd.read_csv(C.PROCESSED_DIR / "tvl_daily.csv", index_col=0, parse_dates=True)
    tvl.index = pd.to_datetime(tvl.index, utc=True)
    orders = pd.read_csv(C.PROCESSED_DIR / "parent_orders.csv")
    orders["ts"] = pd.to_datetime(orders["ts"], utc=True)
    eth_price = json.loads((C.PROCESSED_DIR / "eth_price.json").read_text())["eth_price"]
    gas_gwei = float((C.PROCESSED_DIR / "gas_gwei.txt").read_text().strip())

    chart_paths = {p.stem: str(p) for p in (C.RAW_DIR / "defillama_charts").glob("*.json")}
    meta_full, charts = pool_registry_from_snapshot(
        C.RAW_DIR / "defillama_pools.json", chart_paths, gas_gwei=gas_gwei, eth_price=eth_price
    )
    meta = meta.set_index("name").loc[list(tvl.columns)].reset_index()
    state = build_pool_state(meta, {r["pool_id"]: charts[r["pool_id"]] for _, r in meta.iterrows()}, eth_price)

    print(f"\nReal orders: {len(orders)}   window: {orders['ts'].min().date()} .. {orders['ts'].max().date()}")
    print(f"F: min ${orders['F'].min():,.0f}  median ${orders['F'].median():,.0f}  "
          f"p95 ${orders['F'].quantile(0.95):,.0f}  max ${orders['F'].max():,.0f}")
    print(f"Pools (M={state.M}): {', '.join(state.names)}")

    results = run_backtest(
        orders, state.tvl, state.a, state.b, state.gas_usd,
        alpha=C.ALPHA, lam=C.LAMBDA, eta=C.ETA, verbose=True,
    )
    results.to_csv(C.PROCESSED_DIR / "backtest_results.csv", index=False)

    summary, paired = summarize_backtest(results, alpha=C.ALPHA)
    by_size = backtest_by_size(results, alpha=C.ALPHA)
    bad = bad_day_comparison(results, alpha=C.ALPHA)

    print("\n" + "=" * 70)
    print("REALISED COSTS ON REAL TRADES")
    print("=" * 70)
    cols = ["mean", "median", "std", "max", f"VaR_{C.ALPHA:.2f}", f"CVaR_{C.ALPHA:.2f}", "bps_mean", "bps_CVaR_0.95"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].round(2).to_string())
    summary.round(4).to_csv(C.TABLES_DIR / "backtest_summary.csv")

    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISON: CVaR router vs baselines  (alpha={C.ALPHA})")
    print("=" * 70)
    print(paired.drop(columns=[c for c in ("CVaR_A", "CVaR_B") if c in paired.columns]).round(3).to_string())
    paired.round(4).to_csv(C.TABLES_DIR / "backtest_paired.csv")

    print("\n" + "=" * 70)
    print("STRESSED LIQUIDITY DAYS (top 20% by mean delta_real)")
    print("=" * 70)
    print(f"  {bad.attrs.get('n_bad_days', '?')} days, severity threshold {bad.attrs.get('severity_threshold', float('nan')):.3f}, "
          f"{bad['n'].iloc[0]} orders" if len(bad) else "  (none)")
    print(bad[["mean", "max", f"CVaR_{C.ALPHA:.2f}"]].round(2).to_string())
    bad.round(4).to_csv(C.TABLES_DIR / "backtest_bad_days.csv")

    print("\n" + "=" * 70)
    print("BY ORDER SIZE")
    print("=" * 70)
    piv = by_size.pivot(index="size_bucket", columns="strategy", values=f"CVaR_{C.ALPHA:.2f}")
    print(piv.round(1).to_string())
    by_size.round(4).to_csv(C.TABLES_DIR / "backtest_by_size.csv", index=False)

    # model fidelity figure on the real tape
    impact_df = pd.read_csv(C.PROCESSED_DIR / "impact_sample.csv")
    fit = fit_impact_regression(impact_df)
    (C.TABLES_DIR / "impact_fit.json").write_text(json.dumps(fit, indent=2))
    print("\n" + "=" * 70)
    print("MODEL FIDELITY CHECK (real tape)")
    print("=" * 70)
    print(f"  bin-mean LSQ fit:  cost ~= {fit['a_tape']:.4e} n + {fit['b_tape']:.4e} n^2   R^2={fit['r2']:.4f}")
    print(f"  cost increasing in size: {fit['cost_increasing']}   (rank corr {fit['spearman_size_cost']:.2f})")
    print(f"  cost per USD: {fit['bps_at_smallest_bin']:.2f} bps (smallest bin) -> {fit['bps_at_largest_bin']:.2f} bps (largest bin)")
    print("  NOTE: the quadratic-in-n term is an AMM (xy=k) property derived analytically;")
    print("        the CEX tape confirms positive, size-increasing execution cost and sets")
    print("        the linear fee/spread scale.  Curvature on a CEX book is venue-specific.")
    plots.fig_impact_fit(impact_df, fit)

    plots.fig_backtest(results, C.ALPHA)
    plots.fig_backtest_tail(results, C.ALPHA)
    plots.fig_backtest_by_size(by_size, C.ALPHA)

    print("\nDone. Backtest tables + figures written.")


if __name__ == "__main__":
    main()
