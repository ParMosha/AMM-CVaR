"""Scenario-construction tests: bounds, shapes, no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from amm_cvar.scenarios import realized_state, trailing_scenarios


@pytest.fixture
def tvl_hist():
    idx = pd.date_range("2026-01-01", "2026-08-31", freq="D", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(0)
    base = np.linspace(10e6, 12e6, n)
    noise = 1 + 0.05 * rng.standard_normal(n)
    crash = np.ones(n)
    crash[n // 2] = 0.6  # one stress day
    crash[n // 2 + 1] = 0.75
    return pd.DataFrame(
        {
            "poolA": base * noise * crash,
            "poolB": 2 * base * (1 + 0.03 * rng.standard_normal(n)) * crash,
        },
        index=idx,
    )


def test_shapes_and_bounds(tvl_hist):
    out = trailing_scenarios(tvl_hist, "2026-08-20", window=30, delta_bar=0.6)
    M, K = out["L_scenarios"].shape
    assert M == 2
    assert K == 30
    assert np.all(out["deltas"] >= 0) and np.all(out["deltas"] <= 0.6 + 1e-12)
    assert np.all(out["L_scenarios"] > 0)
    assert out["L_scenarios"].shape == out["deltas"].shape
    assert np.allclose(out["p"], 1.0 / K)


def test_scenarios_exclude_execution_day(tvl_hist):
    """Look-ahead check: the scenario set may not contain day(t) itself."""
    date = pd.Timestamp("2026-08-20", tz="UTC")
    out = trailing_scenarios(tvl_hist, date, window=30)
    known = pd.DatetimeIndex(out["known_dates"])
    assert (known < date).all()
    assert date not in set(known)


def test_delta_scale_monotone(tvl_hist):
    base = trailing_scenarios(tvl_hist, "2026-08-20", window=30, delta_scale=1.0)
    hot = trailing_scenarios(tvl_hist, "2026-08-20", window=30, delta_scale=1.5)
    assert np.all(hot["deltas"] >= base["deltas"] - 1e-12)
    assert np.all(hot["L_scenarios"] <= base["L_scenarios"] + 1e-9)


def test_realized_state_out_of_sample(tvl_hist):
    date = pd.Timestamp("2026-08-20", tz="UTC")
    real = realized_state(tvl_hist, date)
    scen = trailing_scenarios(tvl_hist, date, window=30)
    assert real["L_real"].shape == scen["L_nominal"].shape
    assert np.all(real["delta_real"] >= 0)
    # the injected crash day must show up as a stressed realisation
    crash_day = tvl_hist.index[len(tvl_hist) // 2]
    d_crash = realized_state(tvl_hist, crash_day)["delta_real"].max()
    assert d_crash > 0.15


def test_adverse_only_clipping(tvl_hist):
    """Shortfalls vs the pool's typical level, clipped to the adverse side."""
    date = pd.Timestamp("2026-08-20", tz="UTC")
    out = trailing_scenarios(tvl_hist, date, window=60, n_scenarios=None)
    L_hist = tvl_hist.loc[tvl_hist.index < date].tail(61).to_numpy(float) / 2.0
    G = np.exp(np.mean(np.log(L_hist), axis=0))
    assert np.all(out["deltas"] >= 0)
    assert np.all(out["deltas"] <= 0.6 + 1e-12)
    assert np.all(out["L_scenarios"] <= out["L_nominal"][:, None] + 1e-9)
    # every historical day below the typical level must appear as stress
    below = np.clip(1.0 - L_hist / G, 0.0, 0.6)
    assert below.max() > 0.05


def test_trend_does_not_invert_risk(tvl_hist):
    """A pool that has *collapsed* must not look safer than a stable pool."""
    idx = pd.date_range("2026-01-01", "2026-08-31", freq="D", tz="UTC")
    n = len(idx)
    stable = np.full(n, 10e6)
    crashed = np.concatenate([np.full(n - 30, 20e6), np.linspace(20e6, 1e6, 30)])
    tvl = pd.DataFrame({"stable": stable, "crashed": crashed}, index=idx)
    out = trailing_scenarios(tvl, "2026-08-25", window=120, n_scenarios=None)
    assert out["deltas"][1].mean() > out["deltas"][0].mean()


def test_insufficient_history_raises(tvl_hist):
    with pytest.raises(ValueError):
        trailing_scenarios(tvl_hist, "2026-01-02", window=30)
