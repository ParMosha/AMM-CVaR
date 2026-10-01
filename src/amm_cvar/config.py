"""Central configuration: paths, seeds, windows and model defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Market-data windows (UTC dates)
# ---------------------------------------------------------------------------
SYMBOL = "ETHUSDT"
# 30 consecutive days of real trades used for the backtest / order-size tape.
TRADE_START = "2026-08-16"
TRADE_END = "2026-09-14"
# Trailing TVL history (real DefiLlama daily points) used for scenarios + plots.
TVL_HISTORY_DAYS = 900
# Fallback trade window if the recent one is unavailable offline.
TRADE_START_FALLBACK = "2025-06-01"
TRADE_END_FALLBACK = "2025-06-30"

# ---------------------------------------------------------------------------
# Parent-order construction from the real trade tape
# ---------------------------------------------------------------------------
PARENT_FREQ = "1min"          # aggregate taker sells into 1-minute parent orders
MIN_ORDER_USD = 1_000.0       # ignore dust buckets
MAX_BACKTEST_ORDERS = 2_500   # stratified sample size for the walk-forward loop
IMPACT_SAMPLE_PER_DAY = 4_000 # raw trades kept per day for the impact regression

# ---------------------------------------------------------------------------
# Cost model (USD notional units throughout)
# ---------------------------------------------------------------------------
#   c_m(f_m)  = a_m f_m + b_m f_m^2 / L_m ,   L_m = TVL_m / 2  (USD)
#   a_m       = pool fee rate phi_m  (derived in README from xy=k + fees)
#   b_m       = 1.0                (second-order Taylor coefficient of xy=k impact)
B_M_DEFAULT = 1.0
DELTA_BAR = 0.6               # scenario liquidity degradation cap, delta in [0, DELTA_BAR)

# ---------------------------------------------------------------------------
# Uncertainty / scenarios
# ---------------------------------------------------------------------------
# Historical-simulation lookback: adverse daily TVL deviations of the trailing
# `SCENARIO_WINDOW` days form the candidate scenario set (no look-ahead).
SCENARIO_WINDOW = 365
N_SCENARIOS = 120             # K scenarios kept (worst shortfall days + sample)
N_WORST_SCENARIOS = 24        # ... of which this many populate the CVaR tail
DELTA_SCALE = 1.0             # uncertainty intensity multiplier (swept in sensitivity)

# ---------------------------------------------------------------------------
# Optimization defaults
# ---------------------------------------------------------------------------
ALPHA = 0.95                  # CVaR confidence level
LAMBDA = 1.0                  # weight on CVaR in  mean + lambda * CVaR
ETA = 0.0                     # pool-usage penalty (proved constant at optimum, see README)
SOLVER_PRIMARY = "CLARABEL"
SOLVER_FALLBACK = "SCS"

# ---------------------------------------------------------------------------
# Baseline / evaluation
# ---------------------------------------------------------------------------
ACTIVE_POOL_TOL_FRAC = 0.01   # pool counts as "active" if f_m > 1% of F
GAS_USED_TOL_USD = 0.0        # any positive (non-dusted) allocation pays the pool's gas

# Estimated swap gas (units) per venue; converted to USD with eth_gasPrice * ETH price.
GAS_UNITS = {
    "uniswap-v3": 185_000,
    "uniswap-v2": 150_000,
    "sushiswap": 150_000,
    "curve-dex": 300_000,
}
GAS_GWEI_FALLBACK = 20.0

# Sensitivity grids
ALPHA_GRID = [0.80, 0.90, 0.95, 0.99]
F_GRID_USD = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
DELTA_SCALE_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
SCALABILITY_GRID = [(5, 50), (5, 100), (5, 200), (10, 50), (10, 100), (10, 200), (20, 50), (20, 100), (20, 200)]

# Base case order size used for the in-sample experiments (USD).
BASE_ORDER_USD = 250_000.0


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Return a fresh, reproducible random generator."""
    return np.random.default_rng(seed)
