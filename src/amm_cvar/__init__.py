"""AMM order splitting with CVaR tail-risk management (CO Project 3, rewrite)."""

from .cost_model import (
    scenario_cost_numpy,
    true_execution_cost,
    xyk_exact_impact,
    fit_impact_regression,
)
from .optimizer import solve_cvar_routing, problem_dimensions
from .baselines import (
    single_pool_best_mean,
    single_pool_best_liquidity,
    equal_split,
    liquidity_proportional,
    mean_only_allocation,
)
from .metrics import empirical_var, empirical_cvar, summarize_losses
from .scenarios import trailing_scenarios, scenario_cost_matrix
from .backtest import run_backtest
from .pools import build_pool_state, pool_registry_from_snapshot

__all__ = [
    "scenario_cost_numpy",
    "true_execution_cost",
    "xyk_exact_impact",
    "fit_impact_regression",
    "solve_cvar_routing",
    "problem_dimensions",
    "single_pool_best_mean",
    "single_pool_best_liquidity",
    "equal_split",
    "liquidity_proportional",
    "mean_only_allocation",
    "empirical_var",
    "empirical_cvar",
    "summarize_losses",
    "trailing_scenarios",
    "scenario_cost_matrix",
    "run_backtest",
    "build_pool_state",
    "pool_registry_from_snapshot",
]
