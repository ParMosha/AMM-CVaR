"""Historical scenario generation from *real* per-pool TVL deviations.

The project statement allows any scenario model with

    L_{m,k} = L_m * (1 - delta_{m,k}),   delta_{m,k} in [0, delta_bar).

Definition (adverse historical resampling)
-----------------------------------------
For decision day ``d`` and trailing historical day ``d-k``:

    G_m       = geometric mean of L_m over the trailing window (typical level)
    delta_m,k = clip( 1 - L_m(d-k) / G_m , 0 , delta_bar )

i.e. the shortfall of day ``d-k`` versus pool ``m``'s *own typical level*,
clipped to the adverse side (the assignment only admits degradation,
``delta >= 0``).

Why deviations from the pool's own typical level and not from "today"?
Raw ratios ``1 - L_m(d-k)/L_m(d-1)`` confound *trend* with *risk*: a pool that
has secularly grown looks permanently "risky" (its past is below today) while a
pool that has collapsed looks "safe" (its past is above today).  Measuring
shortfalls against the window's geometric mean is trend-robust and puts the risk
ordering back in line with economic intuition (verified on the real pools in
the report).

The same historical day ``k`` is used for every pool, which preserves
cross-pool correlation (a market-wide liquidity event hits all pools together).
Only data strictly before the decision date is used (no look-ahead).

Scenario count
--------------
``n_scenarios`` days are kept out of the trailing ``window``: the
``n_worst`` days by average shortfall (to populate the CVaR tail) plus a
uniform random sample of the remainder (fixed seed -> reproducible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _to_utc_day(date) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def trailing_scenarios(
    tvl: pd.DataFrame,
    date,
    window: int = C.SCENARIO_WINDOW,
    delta_bar: float = C.DELTA_BAR,
    delta_scale: float = C.DELTA_SCALE,
    n_scenarios: int | None = C.N_SCENARIOS,
    n_worst: int | None = None,
    nominal: str = "last",
    seed: int = C.SEED,
) -> dict:
    """Build (L_nominal, deltas (M,K), L_scenarios (M,K)) for decision ``date``.

    Parameters
    ----------
    tvl : daily TVL matrix (USD), index = UTC dates, columns = pools
    date : decision timestamp
    window : trailing days of real history searched for adverse states
    delta_scale : uncertainty-intensity multiplier (sensitivity sweep)
    n_scenarios : K scenarios kept out of the window (None -> all)
    n_worst : how many of those are the worst shortfall days (default max(10, K//6))
    nominal : 'last' -> L_nominal from the last known day (d-1)
    """
    date = _to_utc_day(date)
    index = tvl.index
    known = index[index < date]          # strictly before the execution day
    if len(known) < 5:
        raise ValueError(f"Insufficient TVL history before {date.date()}")
    hist = tvl.loc[known].tail(window + 1)
    L_hist = hist.to_numpy(float) / 2.0  # (T, M) liquidity index = TVL/2

    if nominal == "date":
        L_nom = tvl.loc[index[index <= date][-1]].to_numpy(float) / 2.0
    else:
        L_nom = L_hist[-1].copy()

    M = L_hist.shape[1]
    T = L_hist.shape[0] - 1               # number of historical days (excl. today)
    K_all = min(window, T)

    # typical level per pool (geometric mean over the trailing window)
    G = np.exp(np.mean(np.log(np.maximum(L_hist, 1e-9)), axis=0))   # (M,)

    # adverse deviations of each trailing day vs the pool's typical level
    deltas_all = np.zeros((M, K_all))
    dates_all = np.empty(K_all, dtype=object)
    for k in range(K_all):
        day_state = L_hist[-(k + 2)]      # k=0 -> the day before the reference day
        deltas_all[:, k] = np.clip(1.0 - day_state / G, 0.0, delta_bar)
        dates_all[k] = known[-(k + 2)]
    deltas_all = np.clip(deltas_all * delta_scale, 0.0, delta_bar)

    # scenario selection: worst shortfall days + uniform sample of the rest
    if n_scenarios is None or n_scenarios >= K_all:
        sel = np.arange(K_all)
    else:
        K = int(n_scenarios)
        n_worst = int(n_worst) if n_worst is not None else C.N_WORST_SCENARIOS
        n_worst = min(n_worst, K)
        severity = deltas_all.mean(axis=0)
        order = np.argsort(severity)[::-1]          # worst first
        worst = order[:n_worst]
        rest = order[n_worst:]
        rng = np.random.default_rng(seed)
        take = min(K - n_worst, len(rest))
        extra = rng.choice(rest, size=take, replace=False) if take > 0 else np.array([], int)
        sel = np.concatenate([worst, extra])
    deltas = deltas_all[:, sel]
    K = deltas.shape[1]

    L_scen = np.maximum(L_nom[:, None] * (1.0 - deltas), 1e-6)
    return {
        "date": date,
        "known_dates": dates_all[sel],
        "L_nominal": L_nom,
        "deltas": deltas,
        "L_scenarios": L_scen,
        "p": np.ones(K) / K,
        "delta_scale": delta_scale,
    }


def realized_state(
    tvl: pd.DataFrame,
    date,
    delta_bar: float = C.DELTA_BAR,
) -> dict:
    """The *actual* liquidity outcome on the execution day (out of sample).

    ``delta_real_m = clip(1 - L_m(day) / G_m, 0, delta_bar)`` with ``G_m`` the
    typical level over the trailing window -- the adverse deviation that
    actually materialised on the execution day.
    """
    date = _to_utc_day(date)
    index = tvl.index
    known = index[index < date]
    if len(known) < 2:
        raise ValueError(f"Insufficient TVL history before {date.date()}")
    L_hist = tvl.loc[known].tail(C.SCENARIO_WINDOW + 1).to_numpy(float) / 2.0
    G = np.exp(np.mean(np.log(np.maximum(L_hist, 1e-9)), axis=0))
    L_last = L_hist[-1].copy()
    day_idx = index[index >= date]
    if len(day_idx) == 0:
        L_real = L_last.copy()
    else:
        L_real = tvl.loc[day_idx[0]].to_numpy(float) / 2.0
    d_real = np.clip(1.0 - L_real / G, 0.0, delta_bar)
    return {"L_real": L_real, "delta_real": d_real, "L_last": L_last, "date": date}


def scenario_cost_matrix(f, L_scenarios, a, b, gas=None, z=None) -> np.ndarray:
    """(K,) costs of allocation ``f`` under the (M, K) liquidity scenarios."""
    from .cost_model import scenario_cost_numpy

    return scenario_cost_numpy(f, L_scenarios, a, b, gas=gas, z=z)
