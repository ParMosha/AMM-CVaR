"""Backtest smoke test on a tiny synthetic tape (no network)."""

import numpy as np
import pandas as pd
import pytest

from amm_cvar.backtest import backtest_by_size, run_backtest, summarize_backtest
from amm_cvar.metrics import empirical_cvar


@pytest.fixture
def tiny_world():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2026-05-01", "2026-08-31", freq="D", tz="UTC")
    n = len(idx)
    base = np.linspace(8e6, 10e6, n)
    tvl = pd.DataFrame(
        {
            "poolA": base * (1 + 0.04 * rng.standard_normal(n)),
            "poolB": 1.5 * base * (1 + 0.04 * rng.standard_normal(n)),
            "poolC": 0.5 * base * (1 + 0.08 * rng.standard_normal(n)),
        },
        index=idx,
    )
    ts = pd.date_range("2026-08-01", "2026-08-20", freq="4h", tz="UTC")
    orders = pd.DataFrame({"ts": ts, "F": rng.uniform(2e4, 3e5, size=len(ts))})
    a = np.array([0.0005, 0.003, 0.003])
    b = np.ones(3)
    gas = np.array([6.0, 5.0, 5.5])
    return {"tvl": tvl, "orders": orders, "a": a, "b": b, "gas": gas}


def test_backtest_smoke(tiny_world):
    w = tiny_world
    res = run_backtest(w["orders"], w["tvl"], w["a"], w["b"], w["gas"],
                       alpha=0.9, lam=1.0, window=30, verbose=False)
    assert len(res) == len(w["orders"]) * 7  # 7 strategies
    assert np.all(np.isfinite(res["realized_cost"]))
    assert set(res["strategy"]) == {
        "CVaR", "CVaR+gasGuard", "MeanOnly", "SinglePoolMean", "SinglePoolDeep", "EqualSplit", "LiqProp",
    }
    # realised cost must respect the budget direction: costs scale with F
    assert res.groupby("strategy")["realized_cost"].mean().min() > 0


def test_backtest_summary_and_sizes(tiny_world):
    w = tiny_world
    res = run_backtest(w["orders"], w["tvl"], w["a"], w["b"], w["gas"], verbose=False)
    summary, paired = summarize_backtest(res, alpha=0.9)
    assert {"mean", "CVaR_0.90"} <= set(summary.columns)
    assert "CVaR" in summary.index and "MeanOnly" in summary.index
    assert (paired["CVaR_improvement_pct"] > -50).all()  # sanity
    sizes = backtest_by_size(res, alpha=0.9)
    assert not sizes.empty
    assert sizes["n"].sum() == len(res)


def test_cvar_router_not_worse_on_tail_than_equal_split(tiny_world):
    w = tiny_world
    res = run_backtest(w["orders"], w["tvl"], w["a"], w["b"], w["gas"], verbose=False)
    pivot = res.pivot_table(index="ts", columns="strategy", values="realized_cost")
    c_cvar = empirical_cvar(pivot["CVaR"].to_numpy(), 0.9)
    c_eq = empirical_cvar(pivot["EqualSplit"].to_numpy(), 0.9)
    # on this tiny world the CVaR router should not be dramatically worse
    assert c_cvar <= c_eq * 1.5
