"""Pool registry and per-pool state built from real DefiLlama TVL history.

Unit convention
---------------
All notionals ``f`` / ``F`` are **USD**.  For a constant-product pool with
balanced reserves worth ``V`` USD, the output-side reserve is ``y = V/2`` and
the liquidity index used by the cost model is

    L = y = V / 2      (USD)

(derivation in ``README.md``).  DefiLlama reports ``tvlUsd = V``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class PoolState:
    """Static + dynamic state of one real pool on a given date."""

    names: list[str]                 # length M
    projects: list[str]              # length M (uniswap-v3, ...)
    fees: np.ndarray                 # phi_m, shape (M,)
    a: np.ndarray                    # linear cost coeff (= fee)
    b: np.ndarray                    # quadratic cost coeff (b_m = 1 by derivation)
    gas_usd: np.ndarray              # g_m, fixed gas cost per swap (estimated)
    tvl: pd.DataFrame                # index=UTC date, columns=pool names, values=USD
    meta: pd.DataFrame               # one row per pool with provenance

    @property
    def M(self) -> int:
        return len(self.names)


def pool_registry_from_snapshot(
    pools_json_path,
    chart_paths: dict[str, str],
    gas_gwei: float | None = None,
    eth_price: float | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the provenance registry + TVL histories from cached downloads."""
    import json

    from .data_sources import load_defillama_chart

    body = json.loads(open(pools_json_path).read())
    rows = body["data"] if isinstance(body, dict) else body
    uni = pd.DataFrame(rows).set_index("pool")

    charts = {pid: load_defillama_chart(p) for pid, p in chart_paths.items()}

    if gas_gwei is None:
        try:
            gas_gwei = float(open(C.PROCESSED_DIR / "gas_gwei.txt").read().strip())
        except Exception:  # noqa: BLE001
            gas_gwei = C.GAS_GWEI_FALLBACK

    records = []
    for pid, chart in charts.items():
        if pid not in uni.index:
            continue
        row = uni.loc[pid]
        project = row["project"]
        meta_fee = row.get("poolMeta")
        fee = _parse_fee(meta_fee, project)
        gas_units = C.GAS_UNITS.get(project, 200_000)
        gas_usd = gas_units * gas_gwei * 1e-9 * (eth_price or 0.0)
        records.append({
            "pool_id": pid,
            "name": _pretty_name(project, row["symbol"], meta_fee),
            "project": project,
            "symbol": row["symbol"],
            "fee": fee,
            "poolMeta": meta_fee,
            "gas_units": gas_units,
            "gas_gwei": gas_gwei,
            "gas_usd_raw": gas_usd,   # needs eth_price if it was 0
            "tvl_current": row["tvlUsd"],
        })
    meta = pd.DataFrame(records).sort_values("tvl_current", ascending=False).reset_index(drop=True)
    return meta, charts


def _parse_fee(meta_fee, project: str) -> float:
    if meta_fee is not None and isinstance(meta_fee, str) and meta_fee.endswith("%"):
        return float(meta_fee[:-1]) / 100.0
    return {"uniswap-v2": 0.003, "sushiswap": 0.003, "curve-dex": 0.0004}.get(project, 0.003)


def _pretty_name(project: str, symbol: str, meta_fee) -> str:
    tag = f" ({meta_fee})" if meta_fee else ""
    return f"{project} {symbol}{tag}"


def align_tvl_series(charts: dict[str, pd.DataFrame], meta: pd.DataFrame) -> pd.DataFrame:
    """Daily TVL (USD) matrix indexed by UTC date, columns = pool names."""
    series = {}
    for _, row in meta.iterrows():
        chart = charts[row["pool_id"]]
        s = chart.set_index("timestamp")["tvlUsd"]
        s.index = s.index.tz_convert("UTC").normalize()
        s = s.groupby(level=0).last()
        series[row["name"]] = s
    tvl = pd.DataFrame(series).sort_index()
    tvl = tvl.ffill().bfill()
    return tvl


def build_pool_state(
    meta: pd.DataFrame,
    charts: dict[str, pd.DataFrame],
    eth_price: float,
    b_default: float = C.B_M_DEFAULT,
) -> PoolState:
    """Assemble the :class:`PoolState` used by the cost model / optimizer."""
    meta = meta.copy()
    gas_units = meta["gas_units"].to_numpy(float)
    gwei = float(meta["gas_gwei"].iloc[0])
    gas_usd = gas_units * gwei * 1e-9 * eth_price

    fees = meta["fee"].to_numpy(float)
    a = fees.copy()          # a_m = phi_m  (see README derivation)
    b = np.full(len(meta), b_default, dtype=float)

    tvl = align_tvl_series(charts, meta)

    meta["gas_usd"] = gas_usd
    meta["a"] = a
    meta["b"] = b
    meta["eth_price_ref"] = eth_price

    return PoolState(
        names=meta["name"].tolist(),
        projects=meta["project"].tolist(),
        fees=fees,
        a=a,
        b=b,
        gas_usd=gas_usd,
        tvl=tvl,
        meta=meta,
    )


def liquidity_from_tvl(tvl_usd: np.ndarray | pd.Series) -> np.ndarray:
    """L_m = TVL_m / 2  (output-side USD reserve of a balanced constant-product pool)."""
    v = np.asarray(tvl_usd, dtype=float)
    return v / 2.0


def pool_state_on(state: PoolState, date) -> dict:
    """Liquidity snapshot known at ``date`` (last TVL point at or before date)."""
    date = pd.Timestamp(date)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")
    else:
        date = date.tz_convert("UTC")
    date = date.normalize()
    idx = state.tvl.index
    eligible = idx[idx <= date]
    if len(eligible) == 0:
        raise ValueError(f"No TVL history at or before {date}")
    row = state.tvl.loc[eligible[-1]]
    L = liquidity_from_tvl(row.to_numpy(float))
    return {
        "date": eligible[-1],
        "tvl": row.to_numpy(float),
        "L": L,
        "a": state.a,
        "b": state.b,
        "gas_usd": state.gas_usd,
    }
