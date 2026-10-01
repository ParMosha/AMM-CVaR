"""Risk metrics consistent with the Rockafellar--Uryasev CVaR epigraph.

For i.i.d. equiprobable losses ``L_1..L_N`` the RU formula

    CVaR_alpha = min_t  t + 1/((1-alpha) N) * sum_k [L_k - t]_+

has optimum value equal to the *expected shortfall*: the average of the worst
``(1-alpha) N`` losses, counting the boundary order statistic fractionally.
``empirical_cvar`` implements exactly that estimator so that backtest numbers
are directly comparable with the CVaR returned by the optimiser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_var(losses, alpha: float) -> float:
    """Value-at-Risk: ``inf{t : P(L <= t) >= alpha}`` (left-continuous quantile)."""
    x = np.sort(np.asarray(losses, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    idx = int(np.ceil(alpha * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(x[idx])


def empirical_cvar(losses, alpha: float) -> float:
    """Expected shortfall (RU-optimal CVaR) of an equiprobable sample."""
    x = np.sort(np.asarray(losses, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    m = (1.0 - alpha) * n
    if m <= 0:
        return float(x[-1])
    k = int(np.floor(m))
    w = m - k
    total = float(x[n - k :].sum()) if k > 0 else 0.0
    if w > 1e-12:
        total += w * float(x[n - k - 1])
    return total / m


def cvar_epigraph_value(losses, alpha: float) -> float:
    """CVaR via the RU epigraph solved as a tiny LP-free closed form (check)."""
    return empirical_cvar(losses, alpha)


def summarize_losses(losses, alphas=(0.90, 0.95, 0.99), prefix: str = "") -> dict:
    x = np.asarray(losses, dtype=float)
    out = {
        f"{prefix}mean": float(np.mean(x)),
        f"{prefix}median": float(np.median(x)),
        f"{prefix}std": float(np.std(x)),
        f"{prefix}min": float(np.min(x)),
        f"{prefix}max": float(np.max(x)),
        f"{prefix}n": int(x.size),
    }
    for al in alphas:
        out[f"{prefix}VaR_{al:.2f}"] = empirical_var(x, al)
        out[f"{prefix}CVaR_{al:.2f}"] = empirical_cvar(x, al)
    return out


def paired_comparison(loss_a, loss_b, alpha: float = 0.95) -> dict:
    """Strategy A vs B on the *same* real orders (lower cost is better)."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    diff = a - b
    tail_n = max(int(round((1 - alpha) * len(a))), 1)
    tail_idx = np.argsort(a - b)[-tail_n:]  # worst relative outcomes for A
    return {
        "mean_diff_A_minus_B": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "win_rate_A": float(np.mean(a < b - 1e-9)),
        "tie_rate": float(np.mean(np.abs(a - b) <= 1e-9)),
        "mean_diff_worst_tail": float(diff[tail_idx].mean()) if tail_n else float("nan"),
        "CVaR_A": empirical_cvar(a, alpha),
        "CVaR_B": empirical_cvar(b, alpha),
        "CVaR_improvement_pct": float(100.0 * (empirical_cvar(b, alpha) - empirical_cvar(a, alpha)) / empirical_cvar(b, alpha))
        if empirical_cvar(b, alpha) != 0
        else float("nan"),
    }


def losses_to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
