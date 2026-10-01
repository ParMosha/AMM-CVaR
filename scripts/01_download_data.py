"""Download real market data into ``data/raw`` (cached).

Usage:
    PYTHONPATH=src python scripts/01_download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone

from amm_cvar import config as C
from amm_cvar import data_sources as ds


def main() -> None:
    print("=" * 70)
    print("01 DOWNLOAD REAL DATA")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Real trade tape (Binance ETHUSDT aggregated trades)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Binance aggTrades {C.SYMBOL}  {C.TRADE_START} .. {C.TRADE_END}")
    manifest = ds.download_aggtrades()
    total_mb = manifest["bytes"].sum() / 1e6
    print(f"      {len(manifest)} daily files, {total_mb:.1f} MB cached in data/raw/aggtrades/")
    if manifest.empty:
        raise SystemExit("No aggTrades downloaded - aborting.")

    days = manifest["date"].tolist()
    print(f"      window: {days[0]} .. {days[-1]}")

    # ------------------------------------------------------------------
    # 2. 1h klines for market context / volatility (real OHLCV)
    # ------------------------------------------------------------------
    start_k = (datetime.now(timezone.utc) - timedelta(days=C.TVL_HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_k = days[-1]
    print(f"\n[2/4] Binance 1h klines {C.SYMBOL}  {start_k} .. {end_k}")
    kman = ds.download_klines(start_k, end_k)
    print(f"      {len(kman)} monthly archives, {kman['bytes'].sum() / 1e6:.2f} MB")

    # ------------------------------------------------------------------
    # 3. Real DEX pool universe snapshot (DefiLlama)
    # ------------------------------------------------------------------
    print("\n[3/4] DefiLlama pool universe snapshot")
    pools_path = ds.download_defillama_pools()
    print(f"      {pools_path}  ({pools_path.stat().st_size / 1e6:.2f} MB)")

    # Select USDC-WETH pools on Ethereum and cache their TVL histories.
    import json

    body = json.loads(pools_path.read_text())
    rows = body["data"] if isinstance(body, dict) else body
    import pandas as pd

    pdf = pd.DataFrame(rows)
    cand = pdf[
        (pdf["chain"] == "Ethereum")
        & (pdf["project"].isin(["uniswap-v3", "uniswap-v2", "sushiswap", "curve-dex"]))
        & (pdf["symbol"].str.contains("USDC"))
        & (pdf["symbol"].str.contains("WETH|ETH", regex=True))
        & (pdf["tvlUsd"] > 1e5)
    ].copy()
    # keep only pairs whose tokens are exactly {USDC, WETH} (allow ordering)
    def _pair_ok(sym: str) -> bool:
        toks = set(sym.split("-"))
        return toks in ({"USDC", "WETH"}, {"USDC", "ETH"})

    cand = cand[cand["symbol"].map(_pair_ok)]
    cand = cand.sort_values("tvlUsd", ascending=False).drop_duplicates(subset=["project", "poolMeta"], keep="first")
    print(f"      candidate USDC-WETH pools: {len(cand)}")
    for _, r in cand.iterrows():
        print(f"        - {r['project']:12s} {r['symbol']:14s} meta={str(r.get('poolMeta')):6s} tvl=${r['tvlUsd']:,.0f}")

    pool_ids = cand["pool"].tolist()
    charts = ds.download_defillama_charts(pool_ids)
    print(f"      cached {len(charts)} TVL histories in data/raw/defillama_charts/")

    # ------------------------------------------------------------------
    # 4. Gas price (optional)
    # ------------------------------------------------------------------
    print("\n[4/4] eth_gasPrice via public RPC")
    gwei = ds.try_eth_gas_price()
    if gwei is None:
        gwei = C.GAS_GWEI_FALLBACK
        print(f"      RPC unavailable -> fallback {gwei:.1f} gwei (documented estimate)")
    else:
        print(f"      {gwei:.2f} gwei")
    (C.PROCESSED_DIR / "gas_gwei.txt").write_text(f"{gwei:.6f}\n")

    print("\nDone. Raw data cached under data/raw/.")


if __name__ == "__main__":
    main()
