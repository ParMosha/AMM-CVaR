"""Downloaders and loaders for *real* market data.

Sources (all public, no API key):
  * Binance Vision  -- daily aggregated trades (real trade tape) and monthly 1h klines
  * DefiLlama       -- real DEX pool TVL history (yields.llama.fi)

Nothing in this module fabricates data.  Any quantity that is estimated
(gas costs, fee constants) is produced by :mod:`amm_cvar.pools` and clearly
labelled there.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.error
import urllib.request

from . import config as C

BINANCE_VISION = "https://data.binance.vision/data/spot"
DEFILLAMA_POOLS = "https://yields.llama.fi/pools"
DEFILLAMA_CHART = "https://yields.llama.fi/chart/{pool_id}"
ETH_RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
]

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_id",
    "last_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_base", "taker_quote", "ignore",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "amm-cvar-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def _http_get_optional(url: str, timeout: int = 30) -> bytes | None:
    try:
        return _http_get(url, timeout=timeout, retries=2)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Binance Vision
# ---------------------------------------------------------------------------
def aggtrade_day_url(symbol: str, day: str) -> str:
    return f"{BINANCE_VISION}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"


def kline_month_url(symbol: str, month: str) -> str:
    return f"{BINANCE_VISION}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"


def _download_one(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    raw = _http_get_optional(url, timeout=120)
    if raw is None or len(raw) < 200:
        return None
    dest.write_bytes(raw)
    return dest


def download_aggtrades(
    start: str = C.TRADE_START,
    end: str = C.TRADE_END,
    symbol: str = C.SYMBOL,
    raw_dir: Path = C.RAW_DIR,
    max_days: int = 30,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Download up to ``max_days`` of real aggregated trades.

    Missing days are replaced by walking backwards from ``end`` so we always
    keep a contiguous-as-possible 30-day tape.  Returns a manifest DataFrame.
    """
    out_dir = raw_dir / "aggtrades" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    # padding candidates in case some recent days are missing
    pad = pd.date_range(end="2027-01-01", periods=1, freq="D")  # placeholder, unused
    del pad
    extra = (
        pd.date_range(start=pd.Timestamp(end) - pd.Timedelta(days=90), end=start, freq="D")
        .strftime("%Y-%m-%d")
        .tolist()[::-1]
    )
    candidates = days + extra

    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for day in candidates:
            dest = out_dir / f"{symbol}-aggTrades-{day}.zip"
            jobs.append(pool.submit(_download_one, aggtrade_day_url(symbol, day), dest))
        for fut in as_completed(jobs):
            fut.result()

    files = sorted(out_dir.glob(f"{symbol}-aggTrades-*.zip"))
    rows = [
        {"date": p.stem.split("aggTrades-")[-1], "path": str(p), "bytes": p.stat().st_size}
        for p in files
    ]
    manifest = pd.DataFrame(rows)
    if len(manifest) > max_days:
        # keep the requested window first, then the closest padding days
        manifest = manifest.sort_values("date").tail(max_days).reset_index(drop=True)
        for p in files:
            day = p.stem.split("aggTrades-")[-1]
            if day not in set(manifest["date"]):
                # keep all downloads (cheap) but mark them
                pass
    manifest = manifest.sort_values("date").reset_index(drop=True)
    manifest.to_csv(C.PROCESSED_DIR / "aggtrades_manifest.csv", index=False)
    return manifest


def download_klines(
    start: str,
    end: str,
    symbol: str = C.SYMBOL,
    raw_dir: Path = C.RAW_DIR,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Download monthly 1h kline archives covering [start, end]."""
    out_dir = raw_dir / "klines1h" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(start, end, freq="M").strftime("%Y-%m").tolist()
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for month in months:
            dest = out_dir / f"{symbol}-1h-{month}.zip"
            jobs.append(pool.submit(_download_one, kline_month_url(symbol, month), dest))
        for fut in as_completed(jobs):
            fut.result()
    files = sorted(out_dir.glob(f"{symbol}-1h-*.zip"))
    manifest = pd.DataFrame(
        [{"month": p.stem.split("1h-")[-1], "path": str(p), "bytes": p.stat().st_size} for p in files]
    )
    manifest.to_csv(C.PROCESSED_DIR / "klines_manifest.csv", index=False)
    return manifest


def download_defillama_pools(raw_dir: Path = C.RAW_DIR) -> Path:
    """Snapshot of the live DefiLlama pool universe (used to pick real pools)."""
    dest = raw_dir / "defillama_pools.json"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.write_bytes(_http_get(DEFILLAMA_POOLS, timeout=60))
    return dest


def download_defillama_charts(pool_ids: list[str], raw_dir: Path = C.RAW_DIR) -> dict[str, Path]:
    """Historical TVL / APY series for the selected real pools."""
    out_dir = raw_dir / "defillama_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for pid in pool_ids:
        dest = out_dir / f"{pid}.json"
        if not (dest.exists() and dest.stat().st_size > 1000):
            raw = _http_get_optional(DEFILLAMA_CHART.format(pool_id=pid), timeout=60)
            if raw is None:
                continue
            dest.write_bytes(raw)
        paths[pid] = dest
    return paths


def try_eth_gas_price() -> float | None:
    """Current eth_gasPrice in gwei via a public RPC (None if unreachable)."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_gasPrice", "params": []}).encode()
    for endpoint in ETH_RPC_ENDPOINTS:
        try:
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "amm-cvar-research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
            hex_wei = body["result"]
            return int(hex_, 16) / 1e9
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def load_aggtrade_zip(path: Path | str) -> pd.DataFrame:
    """Parse one daily aggTrades archive into a typed DataFrame."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=AGGTRADE_COLUMNS)
    df["price"] = df["price"].astype("float64")
    df["quantity"] = df["quantity"].astype("float64")
    df["notional"] = df["price"] * df["quantity"]
    # Binance Vision emits ms timestamps in older archives and us in newer ones.
    ts = df["transact_time"].to_numpy()
    unit = "us" if (len(ts) and float(np.nanmax(ts)) > 1e14) else "ms"
    df["transact_time"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    return df


def load_kline_zip(path: Path | str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=KLINE_COLUMNS)
    ts = df["open_time"].to_numpy()
    unit = "us" if (len(ts) and float(np.nanmax(ts)) > 1e14) else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"):
        df[col] = df[col].astype("float64")
    return df


def load_defillama_chart(path: Path | str) -> pd.DataFrame:
    body = json.loads(Path(path).read_text())
    rows = body.get("data", body)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Trade-tape feature engineering
# ---------------------------------------------------------------------------
def parent_orders_from_trades(
    trades: pd.DataFrame,
    freq: str = C.PARENT_FREQ,
    min_notional: float = C.MIN_ORDER_USD,
) -> pd.DataFrame:
    """Aggregate *real* taker sells into parent orders of notional ``F`` (USD).

    ``is_buyer_maker == True`` marks an aggressive sell (taker is the seller),
    which matches the modelled direction (sell ETH for USDC).
    """
    sells = trades.loc[trades["is_buyer_maker"]].copy()
    if sells.empty:
        return pd.DataFrame(columns=["ts", "F", "n_trades", "vwap"])
    sells = sells.set_index("transact_time").sort_index()
    grouped = sells.resample(freq)
    orders = pd.DataFrame({
        "F": grouped["notional"].sum(),
        "n_trades": grouped["agg_trade_id"].count(),
        "base_qty": grouped["quantity"].sum(),
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        orders["vwap"] = np.where(orders["base_qty"] > 0, orders["F"] / orders["base_qty"], np.nan)
    orders = orders.dropna(subset=["F"])
    orders = orders[orders["F"] >= min_notional].copy()
    orders.index.name = "ts"
    return orders.reset_index()


def sample_trades_for_impact(trades: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Uniform sample of raw trades used for the impact-shape regression."""
    if len(trades) <= n:
        return trades.copy()
    idx = rng.choice(len(trades), size=n, replace=False)
    return trades.iloc[np.sort(idx)].copy()


def realized_impact_1s(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-second realised price impact of aggressive sells (model fidelity check).

    cost_usd  ~ (pre-trade mid - vwap) / pre-trade mid * notional

    Aggregated to 1s buckets to reduce bid/ask bounce; the caller bins these
    observations by size before fitting (see ``cost_model.fit_impact_regression``).
    """
    df = trades.sort_values("transact_time").copy()
    df["second"] = df["transact_time"].dt.floor("1s")
    df["pre_mid"] = df["price"].shift(1).bfill()
    sells = df.loc[df["is_buyer_maker"]].copy()
    if sells.empty:
        return pd.DataFrame(columns=["notional", "cost_usd"])
    g = sells.groupby("second")
    out = pd.DataFrame({
        "notional": g["notional"].sum(),
        "base_qty": g["quantity"].sum(),
        "pre_mid": g["pre_mid"].first(),
    })
    out["vwap"] = out["notional"] / out["base_qty"]
    out["cost_usd"] = (out["pre_mid"] - out["vwap"]) / out["pre_mid"] * out["notional"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["notional"] > 0]
    return out[["notional", "cost_usd", "pre_mid", "vwap"]].reset_index(drop=True)
