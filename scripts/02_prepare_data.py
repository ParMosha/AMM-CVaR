"""Prepare analysis inputs from cached raw data.

Builds (all in data/processed):
  * pools_meta.csv / tvl_daily.csv      -- real pool registry + real TVL
  * parent_orders.csv                   -- real 1-min taker-sell parent orders
  * impact_sample.csv                   -- 1s realised impacts (model fidelity)
  * impact_fit.json                     -- least-squares impact shape on the tape
  * trade_stats.json                    -- descriptive stats of the real tape
  * eth_price.json / gas_usd.json       -- unit conversion for gas

Usage:
    PYTHONPATH=src python scripts/02_prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amm_cvar import config as C
from amm_cvar import data_sources as ds
from amm_cvar import pools as pools_mod
from amm_cvar.cost_model import fit_impact_regression


def main() -> None:
    print("=" * 70)
    print("02 PREPARE REAL DATA")
    print("=" * 70)

    rng = C.set_seed(C.SEED)

    # ------------------------------------------------------------------
    # 1. Trade tape -> parent orders + impact sample
    # ------------------------------------------------------------------
    manifest = pd.read_csv(C.PROCESSED_DIR / "aggtrades_manifest.csv")
    all_orders = []
    impact_chunks = []
    n_trades = 0
    sell_notional = 0.0
    buy_notional = 0.0
    size_sample = []

    for i, row in manifest.iterrows():
        trades = ds.load_aggtrade_zip(row["path"])
        n_trades += len(trades)
        sell = trades.loc[trades["is_buyer_maker"], "notional"]
        buy = trades.loc[~trades["is_buyer_maker"], "notional"]
        sell_notional += float(sell.sum())
        buy_notional += float(buy.sum())
        size_sample.append(trades["notional"].to_numpy())

        orders = ds.parent_orders_from_trades(trades)
        if not orders.empty:
            all_orders.append(orders)

        impact_chunks.append(ds.realized_impact_1s(ds.sample_trades_for_impact(trades, C.IMPACT_SAMPLE_PER_DAY, rng)))
        print(f"  [{i + 1:02d}/{len(manifest)}] {row['date']}: {len(trades):,} trades, "
              f"{len(orders)} parent orders, F_total=${orders['F'].sum() if len(orders) else 0:,.0f}")

    orders_df = pd.concat(all_orders, ignore_index=True)
    orders_df["ts"] = pd.to_datetime(orders_df["ts"], utc=True)
    orders_df = orders_df.sort_values("ts").reset_index(drop=True)

    # stratified sample across time x size for the walk-forward loop
    if len(orders_df) > C.MAX_BACKTEST_ORDERS:
        orders_df = orders_df.assign(
            _bin=lambda d: pd.qcut(d["F"], q=10, duplicates="drop")
        )
        picked = (
            orders_df.groupby("_bin", group_keys=False, observed=True)
            .apply(lambda g: g.sample(min(len(g), max(C.MAX_BACKTEST_ORDERS // orders_df["_bin"].nunique(), 1)), random_state=C.SEED))
            .drop(columns="_bin")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        orders_df = picked

    orders_df.to_csv(C.PROCESSED_DIR / "parent_orders.csv", index=False)
    print(f"\n  parent orders kept: {len(orders_df)}  "
          f"(F: median ${orders_df['F'].median():,.0f}, p95 ${orders_df['F'].quantile(0.95):,.0f}, "
          f"max ${orders_df['F'].max():,.0f})")

    sizes = np.concatenate(size_sample)
    trade_stats = {
        "n_trades": int(n_trades),
        "sell_notional_usd": sell_notional,
        "buy_notional_usd": buy_notional,
        "median_trade_usd": float(np.median(sizes)),
        "p95_trade_usd": float(np.quantile(sizes, 0.95)),
        "p999_trade_usd": float(np.quantile(sizes, 0.999)),
        "mean_trade_usd": float(np.mean(sizes)),
    }
    (C.PROCESSED_DIR / "trade_stats.json").write_text(json.dumps(trade_stats, indent=2))
    print(f"  real trades parsed: {n_trades:,}  (notional traded ${sell_notional + buy_notional:,.0f})")

    # ------------------------------------------------------------------
    # 2. Impact-shape regression on the real tape
    # ------------------------------------------------------------------
    impact_df = pd.concat(impact_chunks, ignore_index=True)
    impact_df.to_csv(C.PROCESSED_DIR / "impact_sample.csv", index=False)
    fit = fit_impact_regression(impact_df)
    (C.PROCESSED_DIR / "impact_fit.json").write_text(json.dumps(fit, indent=2))
    print(f"\n  impact fit on real tape: cost ~= {fit['a_tape']:.3e} n + {fit['b_tape']:.3e} n^2   "
          f"(R2={fit['r2']:.3f}, n={fit['n_obs']:,})")

    # ------------------------------------------------------------------
    # 3. Real pools + TVL
    # ------------------------------------------------------------------
    gas_gwei = float((C.PROCESSED_DIR / "gas_gwei.txt").read_text().strip())
    kman = pd.read_csv(C.PROCESSED_DIR / "klines_manifest.csv")
    klines = pd.concat([ds.load_kline_zip(p) for p in kman["path"]], ignore_index=True).sort_values("open_time")
    eth_price = float(klines["close"].iloc[-1])
    (C.PROCESSED_DIR / "eth_price.json").write_text(json.dumps({"eth_price": eth_price, "ts": str(klines["open_time"].iloc[-1])}))
    print(f"\n  ETH price ref (last 1h close): ${eth_price:,.2f}   gas: {gas_gwei:.2f} gwei")

    chart_paths = {p.stem: str(p) for p in (C.RAW_DIR / "defillama_charts").glob("*.json")}
    meta, charts = pools_mod.pool_registry_from_snapshot(
        C.RAW_DIR / "defillama_pools.json", chart_paths, gas_gwei=gas_gwei, eth_price=eth_price
    )
    tvl = pools_mod.align_tvl_series(charts, meta)

    # keep only pools with real history covering the trade window
    need = pd.Timestamp(C.TRADE_START, tz="UTC") - pd.Timedelta(days=C.SCENARIO_WINDOW + 5)
    ok = [name for name in tvl.columns if tvl[name].loc[:need].notna().sum() > C.SCENARIO_WINDOW]
    meta = meta[meta["name"].isin(ok)].reset_index(drop=True)
    tvl = tvl[ok]
    # drop pools that never had meaningful liquidity in the window
    keep = [n for n in tvl.columns if tvl[n].median() > 2e5]
    meta = meta[meta["name"].isin(keep)].reset_index(drop=True)
    tvl = tvl[keep]

    # rank by mean TVL over the trade window and cap at 6 pools (heterogeneous venues)
    window_tvl = tvl.loc[C.TRADE_START:C.TRADE_END].mean().sort_values(ascending=False)
    top = list(window_tvl.index[:6])
    meta = meta.set_index("name").loc[top].reset_index()
    tvl = tvl[top]

    meta.to_csv(C.PROCESSED_DIR / "pools_meta.csv", index=False)
    tvl.to_csv(C.PROCESSED_DIR / "tvl_daily.csv")
    print(f"\n  selected {len(meta)} real pools:")
    for _, r in meta.iterrows():
        g = r["gas_usd"] if "gas_usd" in r else r["gas_usd_raw"]
        print(f"    - {r['name']:40s} fee={r['fee']*100:.3f}%  gas=${g:.2f}  "
              f"mean TVL=${tvl[r['name']].mean():,.0f}")

    # scenario deltas snapshot (for the data-overview figure)
    scen = __import__("amm_cvar.scenarios", fromlist=["trailing_scenarios"]).trailing_scenarios(
        tvl, pd.Timestamp(C.TRADE_END, tz="UTC")
    )
    np.save(C.PROCESSED_DIR / "delta_hist.npy", scen["deltas"])

    print("\nDone. Processed data in data/processed/.")


if __name__ == "__main__":
    main()
