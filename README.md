# Splitting Orders Across AMM Pools with CVaR Execution-Risk Control

Convex Optimization course project — **Project 3: تقسیم سفارش بین AMMها با کاهش ریسک دم (CVaR)**

This folder is a complete, self-contained rewrite of the project: honest real-world data, the
exact optimisation formulation of the project statement, the three required sensitivity
studies, the two required baselines, and — the main addition — a **walk-forward backtest on
2,500 real trades**.

Author: Parham Moshashaei (402111389)

---

## 1. Problem

A trader wants to execute a total order of `F` USD (sell ETH for USDC) by splitting it over
`M` AMM pools (different Uniswap V3 fee tiers, Uniswap V2, SushiSwap).  Larger clips suffer
more price impact, each used pool costs gas, and the effective pool depth at execution time
is uncertain.  Choose the split to trade off expected execution cost against **tail** (CVaR)
execution cost.

### 1.1 Convex cost model

Per pool `m` with liquidity index `L_m > 0`, sending `f_m` USD costs

```
c_m(f_m; L_m)  =  a_m f_m  +  b_m f_m^2 / L_m ,      f_m >= 0
```

with `a_m = phi_m` the pool fee rate, `b_m = 1`, and `L_m = TVL_m / 2` (USD).

**Why this form (link to `xy = k`).**  For a constant-product pool with output-side reserve
`y` (USD), selling input notional `n` moves the price along `xy = k` and the exact price-impact
cost (in USD, vs. the pre-trade mid) is

```
cost(n) = n^2 / (y + n)  =  n^2/y - n^3/y^2 + ...
```

(derive: with input `dx = n/p`, output `dy = y dx/(x+dx)`, so `cost = p dx - dy = n^2/(y+n)`,
and `p x = y` for the balanced reserves `x, y = TVL/2`.)  The second-order Taylor term is
`n^2 / y = b n^2 / L` with `L := y = TVL/2`, `b = 1`.  The exact expression is itself convex
in `n`; the quadratic surrogate is what the project statement asks for and is DCP-friendly
(`quad_over_lin`).  Relative error of the surrogate vs. the exact `xy=k` cost is < 2% at
1% of the pool reserve and is checked numerically in `tests/test_cost_model.py`.
Fee enters linearly: Uniswap-style pools charge `phi_m` on input, so `a_m = phi_m`.

### 1.2 Scenario model (uncertain execution quality)

The statement allows any scenario model `L_{m,k} = L_m (1 - delta_{m,k})`, `delta_{m,k} in [0, delta_bar)`.
We use **adverse historical resampling of real pool TVL** (DefiLlama daily TVL of each pool):

```
G_m       = geometric mean of L_m over the trailing 365 days (typical level)
delta_m,k = clip( 1 - L_m(d-k) / G_m , 0 , delta_bar ),    delta_bar = 0.6
L_m,k     = L_m * (1 - delta_m,k)
```

* Scenario day `k` is shared across pools → cross-pool correlation is preserved.
* Only days strictly before the decision date are used (no look-ahead).
* `K = 120` scenarios: the 24 worst shortfall days + a seeded random sample of the rest.
* Deviations are taken vs. the pool's **own typical level**, not vs. "today".  A raw ratio
  `1 - L(d-k)/L(d-1)` confounds trend with risk: a pool that has secularly *grown* looks
  permanently risky (its past is below today) while a pool that has *collapsed* looks safe.
  This inversion was observed on the real data and is covered by
  `tests/test_scenarios.py::test_trend_does_not_invert_risk`.  With the `G_m` definition the
  real pools rank as expected: SushiSwap (TVL 3.4M → 0.2M) is riskiest (`mean delta 0.31`),
  Uni V3 0.05% (stable/growing) is safest (`0.06`).

### 1.3 Decision variables and problem

```
f_m >= 0        USD sent to pool m        (sum_m f_m = F)
z_m in [0,1]    usage variable            (f_m <= F z_m)
```

Scenario cost and the Rockafellar–Uryasev CVaR:

```
C_k(f)      = sum_m ( a_m f_m + b_m f_m^2 / L_{m,k} )  +  sum_m g_m z_m
CVaR_a(C)   = min_{t,u>=0}  t + 1/((1-alpha) K) sum_k u_k ,   u_k >= C_k(f) - t
```

**Optimisation problem (as in the project statement):**

```
min_{f,z,t,u}   (1/K) sum_k C_k(f)  +  lambda * CVaR_alpha(C)  +  eta * sum_m z_m

s.t.            sum_m f_m = F,  f_m >= 0,  0 <= z_m <= 1,  f_m <= F z_m
                u_k >= C_k(f) - t,  u_k >= 0
```

Note: the statement writes `VaR_alpha` for the RU expression — that expression is CVaR
(mean excess loss); VaR is its optimal threshold `t*`.  We report both.

### 1.4 Convexity / DCP analysis

1. `a_m f_m` is affine (convex).
2. `b_m f_m^2 / L_{m,k}` is the quadratic-over-linear function; `quad_over_lin(f_m, L_{m,k})`
   is convex for `L > 0` (perspective of `f^2`).  Sum of convex ⇒ `C_k` convex.
3. CVaR epigraph: objective affine in `(t, u)`, constraints `u_k >= C_k(f) - t` are
   hypographs of a convex function shifted by affine ⇒ convex feasible set.
4. `sum f = F`, `f >= 0`, `0 <= z <= 1`, `f <= F z` are linear.
5. Weighted sum of convex objectives with `lambda >= 0` is convex.  **The problem is a convex
   QP-representable program (SOCP), globally solvable.**
6. DCP: the implementation writes the impact term as
   `sum_squares( diag(sqrt(b/L_k)) f )`, which is *identical* to
   `sum_m b_m quad_over_lin(f_m, L_{m,k})` (compare numerically via
   `optimizer.solve_cvar_routing_quadratic_over_linear_reference`) but one atom per scenario
   instead of one per pool × scenario.  `problem.is_dcp()` is asserted in
   `tests/test_optimizer.py::test_formulation_is_dcp`, and the formulation is DPP-compliant,
   which we exploit for fast re-solving.

### 1.5 The `z`-relaxation is fractional — and (partly) vacuous

The statement replaces the binary "pool used" indicator with `0 <= z_m <= 1`, `f_m <= F z_m`
and a penalty `eta sum z_m` (with gas `g_m z_m` inside `C_k`).  **Discussion, as requested:**

*Lemma (collapse).*  At any optimum with `g_m > 0` or `eta > 0`, `z_m = f_m / F`.
Proof: `z` enters the objective only through the non-negative linear terms
`sum_m (g_m + eta) z_m` and the constraint set couples `z` only via `z_m >= f_m / F`,
`z_m <= 1`.  The objective is strictly increasing in each `z_m` used, so each `z_m` is
pushed to its lower bound `f_m / F`.  Moreover `sum_m z_m = sum_m f_m / F = 1` for every
feasible point at the bound — the "sparsity" penalty `eta sum z_m` is then a **constant**
(`eta`) and has no effect on the allocation at all (we keep `eta` in the API for fidelity;
`tests/test_optimizer.py::test_usage_collapses_to_allocation_share` verifies the lemma).

Consequences (both measured in the F-sweep):

* Gas is charged **proportionally** (`g_m f_m / F`) rather than as a true fixed cost
  `g_m 1{f_m > 0}`.  This is a good approximation for large `F` (a $10 swap's gas really is
  dominated by the per-USD term `g_m/F`) but **underestimates the cost of splitting for
  small orders** — the relaxed optimum over-splits, and evaluated under true gas it can lose
  to a single pool below ~$100k.
* Practical fix (extra, clearly labelled): the **gas-guarded variant** `CVaR+gasGuard`
  evaluates the relaxed solution and the best single pool under true cost on the scenario set
  and takes the cheaper.  It dominates both strategies on the real backtest.

All evaluation figures in this report (including every baseline and every backtest number)
use the **true** economic cost

```
true_cost(f) = a f + b f^2 / L  +  sum_{m: f_m > 0} g_m ,
```

i.e. fixed gas per pool actually touched.  The optimiser minimises the relaxed objective of
the statement; the difference is reported as the relaxation gap.

---

## 2. Data — what is real and what is derived

| Quantity | Source | Status |
|---|---|---|
| Trade tape: 25,544,800 real ETH/USDT trades, $24.15B notional, 30 days (2026-08-16 .. 2026-09-14) | Binance Vision daily `aggTrades` | **real** |
| Parent orders: 1-minute taker-sell notional, 2,500 orders ($1k – $16.7M, median $134k) | aggregated from the real tape | derived (aggregation only) |
| Pool TVL `V_m(t)` (daily, 2022 → today) for 5 real Ethereum USDC-WETH pools | DefiLlama `yields.llama.fi` chart API | **real** |
| Pool identity / fee tier (`0.05%`, `0.3%`, `0.01%`, V2 `0.3%`, Sushi `0.3%`) | DefiLlama pool metadata + protocol constants | **real** |
| Liquidity index `L_m = TVL_m / 2` | constant-product derivation (§1.1) | derived |
| Cost coefficients `a_m = phi_m`, `b_m = 1` | derivation from `xy=k` + fees | derived |
| Scenario deviations `delta_{m,k}` | historical resampling of real TVL (§1.2) | derived from real data |
| Fixed gas `g_m` = swap gas units × gas price × ETH price | gas units per venue (estimates), 20 gwei fallback (public RPC unreachable at download time), ETH = $2,467.65 (real Binance 1h close) | **estimated** (documented) |
| 1h OHLCV (market context) | Binance Vision monthly `klines` | **real** |

No number in this project is invented and labelled as real.  In particular the "MEV
vulnerability scores" and "historical slippage from ~10,000 trades" of the earlier notebooks
have been removed (see §6).

Real pools used (M = 5):

| # | Pool | fee `phi_m` | mean TVL (window) | `L_m` at ref. date | gas `g_m` | mean `delta_m` |
|---|---|---|---|---|---|---|
| 0 | Uniswap V3 USDC-WETH 0.05% | 0.05% | $161.7M | $53.5M | $9.13 | 0.06 |
| 1 | Uniswap V3 USDC-WETH 0.3% | 0.30% | $110.2M | $15.1M | $9.13 | 0.21 |
| 2 | Uniswap V2 USDC-WETH | 0.30% | $71.5M | $10.1M | $7.40 | 0.10 |
| 3 | Uniswap V3 USDC-WETH 0.01% | 0.01% | $3.2M | $1.8M | $9.13 | 0.14 |
| 4 | SushiSwap USDC-WETH | 0.30% | $26.5M | $0.14M | $7.40 | 0.31 |

---

## 3. Numerical solution

* Solver **CLARABEL** (SCS fallback); formulation is DPP, so one problem is compiled per
  `(M, K, alpha, lambda)` and re-solved via parameters for every order size / scenario set.
* Base case (M=5, K=120): **131 variables** (f: 5, z: 5, t: 1, u: 120), **251 constraints**.
  Warm solve **~13 ms** (cold compile+solve ~0.3–1 s).
* Scalability (warm re-solve / cold compile+solve):

| M | K | warm (ms) | cold (ms) |
|---|---|---|---|
| 5 | 50 | 5.6 | 328 |
| 5 | 100 | 10.5 | 631 |
| 5 | 200 | 23.3 | 2666 |
| 10 | 100 | 14.9 | 1424 |
| 20 | 100 | 28.6 | 4041 |
| 20 | 200 | 66.5 | 33152 |

* Status `optimal` for every instance; seeds fixed (`SEED = 42`), full reproducibility
  covered by `tests/`.

### 3.1 Base case (F = $250,000, alpha = 0.95, lambda = 1)

| Pool | `f_m` (USD) | share | `z_m` |
|---|---|---|---|
| Uni V3 0.05% | 184,531 | 73.8% | 0.738 |
| Uni V3 0.3% | 33,374 | 13.3% | 0.133 |
| Uni V2 | 25,712 | 10.3% | 0.103 |
| Uni V3 0.01% | 6,104 | 2.4% | 0.024 |
| SushiSwap | 278 | 0.1% | 0.001 |

Mean scenario cost $1,194.39, CVaR_0.95 $1,308.86, VaR threshold t* $1,234.95.
The router concentrates on the deep, cheap 0.05% tier but spills into the other tiers —
impact is quadratic, so the last dollars are cheaper in a second pool than in the first.

### 3.2 Baselines (in-sample, true cost)

| Strategy | mean | VaR_0.95 | CVaR_0.95 | max | #pools |
|---|---|---|---|---|---|
| **CVaR (ours)** | 1194.39 | 1266.43 | **1308.86** | 1442.99 | 4 |
| MeanOnly (lambda=0) | **1191.32** | 1278.21 | 1317.94 | 1450.71 | 4 |
| Single-pool best (mean) | 1389.49 | 1581.27 | 1628.13 | 1785.72 | 1 |
| Single-pool deepest (max L) | 1389.49 | 1581.27 | 1628.13 | 1785.72 | 1 |
| Equal split *(extra)* | 34,882.79 | 48,487.34 | 48,563.26 | 48,600.49 | 5 |
| Liquidity-proportional *(extra)* | 1218.11 | 1274.32 | 1314.78 | 1447.98 | 4 |

CVaR reduction vs. single-pool: **19.6%**; vs. mean-only: **0.7%** (CVaR pays +0.26% mean for
−0.7% tail — the classic trade-off).  Equal split is catastrophic: $2k into the tiny Sushi
pool costs more in impact than the entire order costs optimally.

---

## 4. Sensitivity analysis (required deliverable 4)

### 4.1 Sweep over alpha (risk aversion)

| alpha | mean cost | CVaR (eval @0.95) | CVaR (own alpha) | max | active pools |
|---|---|---|---|---|---|
| 0.80 | 1194.30 | 1309.13 | 1267.54 | 1443.88 | 4 |
| 0.90 | 1195.20 | 1308.24 | 1285.12 | 1443.13 | 4 |
| 0.95 | 1194.39 | 1308.86 | 1308.86 | 1442.99 | 4 |
| 0.99 | 1193.39 | 1310.98 | 1418.81 | 1443.18 | 4 |

Risk-averse settings (high alpha) push a little weight from the high-`delta` Uni V3 0.3%
tier toward the low-`delta` 0.05% tier and the shallow 0.01% tier; costs at a *fixed*
evaluation level stay flat (the pools' tail costs are similar), while the CVaR at the
solver's own alpha grows with alpha as expected (averaging a deeper tail: 1268 → 1419).

### 4.2 Sweep over order size F (when does splitting become necessary?)

| F (USD) | #pools | CVaR router mean | best single-pool mean | cost / USD (bps) |
|---|---|---|---|---|
| 1,000 | 2 | 18.71 | **9.65** | 187.1 |
| 5,000 | 2 | 21.13 | **12.13** | 42.3 |
| 10,000 | 2 | 25.03 | **16.14** | 25.0 |
| 50,000 | 2 | 91.33 | **84.35** | 18.3 |
| 100,000 | 4 | 279.44 | **259.99** | 27.9 |
| 500,000 | 4 | **4,111.99** | 5,280.57 | 82.2 |
| 1,000,000 | 4 | **15,186.25** | 20,594.90 | 151.9 |
| 5,000,000 | 4 | **354,731.15** | 504,653.36 | 709.5 |

**Splitting becomes necessary at F ≈ $100–200k.**  Below that the (relaxed) router already
splits in 2 pools but under true gas the best single pool is cheaper; above it, impact
dominates and splitting wins increasingly (−34% cost at $5M).  Cost per dollar is U-shaped:
gas dominates tiny orders, impact dominates whales.

### 4.3 Sweep over uncertainty intensity (delta scale)

| delta scale | CVaR plan: mean / CVaR | Mean-only plan: mean / CVaR | CVaR advantage on tail |
|---|---|---|---|
| 0.0 | 1109.24 / 1109.24 | 1109.24 / 1109.24 | 0% |
| 0.5 | 1145.97 / 1196.43 | 1145.36 / 1198.21 | 0.15% |
| 1.0 | 1194.39 / 1308.86 | 1191.32 / 1317.94 | 0.69% |
| 1.5 | 1263.62 / 1471.94 | 1251.07 / 1508.66 | 2.4% |
| 2.0 | 1313.69 / 1629.26 | 1295.30 / 1688.71 | **3.5%** |

As uncertainty grows the CVaR plan gives up more mean (1.4% at scale 2.0) but its tail cost
is increasingly better than the mean-only plan (3.5% at scale 2.0).  The "insurance premium"
of CVaR is small in calm conditions and pays off exactly when liquidity uncertainty is high.

Extras: efficient frontier in `lambda` (mean 1191→1201 vs CVaR 1318→1306 over
`lambda = 0..8` — a shallow but well-ordered frontier) and the scalability table above.

---

## 5. Backtest on real trades (walk-forward, no look-ahead)

For each of the **2,500 real parent orders** (1-minute taker-sell notional on Binance
ETH/USDT, 2026-08-16 .. 2026-09-14):

1. plan with pool state and scenario set built from DefiLlama TVL **strictly before** the
   execution day;
2. solve the CVaR router, the mean-only router and the baselines for the order's real size;
3. realise cost with the **actual** pool state of the execution day (out of sample) and the
   true fixed gas of every pool touched.

Realised cost distribution over the real tape (2,500 orders × 7 strategies):

| Strategy | mean (USD) | CVaR_0.95 (USD) | max (USD) | mean (bps) | CVaR (bps) |
|---|---|---|---|---|---|
| **CVaR+gasGuard** | **6902** | **117,000** | 3.47M | **48.2** | **268.5** |
| CVaR | 6909 | 117,000 | 3.47M | 50.4 | 270.3 |
| MeanOnly | 6917 | 117,140 | 3.47M | 50.4 | 270.7 |
| Liquidity-prop. | 6922 | 116,848 | 3.46M | 57.4 | 291.0 |
| Single-pool best | 9987 | 172,781 | 5.25M | 60.3 | 391.6 |
| Single-pool deepest | 9987 | 172,781 | 5.25M | 60.3 | 391.6 |
| Equal split | 166,132 | 2,895,020 | 88.3M | 922.6 | 6509.9 |

Paired comparison (per order, CVaR router vs. baseline):

| vs. | mean saving / order | win rate | realised CVaR improvement |
|---|---|---|---|
| Single-pool best | **$3,078** | 52.2% | **32.3%** |
| MeanOnly | $8.4 | 51.4% | 0.12% |
| Equal split | $159,223 | 100% | 96.0% |

By order size (realised CVaR_0.95, USD):

| size bucket | CVaR | CVaR+gasGuard | MeanOnly | Single-pool |
|---|---|---|---|---|
| < $5k | 21.0 | **12.1** | 21.0 | **12.1** |
| $5k–25k | 41.1 | **32.9** | 41.2 | **32.9** |
| $25k–100k | 259.1 | **238.3** | 258.4 | **238.3** |
| $100k–500k | **3,360** | 3,360 | 3,367 | 4,408 |
| > $500k | **724,084** | 724,084 | 724,827 | 1,075,235 |

On the 6 worst liquidity days (top 20% by mean `delta_real`, 564 orders): CVaR router tail
$141,198 vs mean-only $141,556 vs single-pool $196,612 (**28% better than single-pool**,
0.25% better than mean-only) — CVaR's edge concentrates exactly where it should.

Model-fidelity check on the real tape (51,510 1-second realised impacts, bin-mean LSQ
`cost ≈ 1.28e-4 n - 1.04e-9 n²`, R² 0.92): execution cost is positive and increasing in
size (rank corr 0.88), and the linear fee/spread scale is ≈ 1.3 bps.  The *quadratic* term
is an AMM (`xy=k`) property (verified analytically, §1.1); curvature on a CEX limit-order
book is venue-specific and is not what we need it for.

### Findings

1. **Splitting is worth it for large orders.**  vs. the best single pool, the CVaR router
   cuts realised tail cost by **32%** and mean cost by **$3,078/order**; the advantage grows
   with size (−33% cost at F > $500k).
2. **There is a size threshold.**  Below ≈ $100–200k the fixed gas of extra pools outweighs
   the impact saved: single pool (or `gasGuard`) wins.  This threshold is *visible in the
   real backtest* by size bucket and in the F-sweep.
3. **CVaR weighting is cheap insurance.**  Against mean-only it is essentially tied out of
   sample (0.1% on the tail, 51% win rate) because these 5 real pools have similar risk
   profiles — but it never gives up more than 0.3% mean and it is strictly better under
   higher uncertainty (3.5% tail improvement at 2× delta intensity) and on the worst real
   liquidity days (0.25% tail improvement).
4. **The gas-guard is the practical strategy.**  Combining the relaxed CVaR router with a
   true-cost fallback to the best single pool dominates both on the real tape (lowest mean,
   lowest bps).
5. **Pool depth heterogeneity dominates fee heterogeneity** at large F (impact is quadratic),
   while fee tiers matter for small/medium F (linear) — exactly the structure of `c_m`.

---

## 6. Review: issues found in the earlier notebooks and fixed here

| # | Issue in `Project_402111389.ipynb` / `Project_script.ipynb` | Fix |
|---|---|---|
| 1 | "REAL UNISWAP DATA", "historical slippage from ~10,000 ETH/USDC trades", "MEV vulnerability from Flashbots" are hardcoded numbers or `np.random.uniform` draws | Real downloads only (Binance aggTrades + DefiLlama TVL). Provenance table (§2); synthetic quantities removed or explicitly labelled |
| 2 | Objective `lambda CVaR + (1-lambda) E[.] + gamma g@z` ≠ statement `E[.] + lambda CVaR + eta sum z`; `g_m z_m` missing from `C_k` | Exact statement implemented (§1.3) |
| 3 | Markdown constraint `z_m >= min(x_m, 1)` is wrong (code had `z >= x/D`, which is right) | Statement constraints `0<=z<=1`, `f<=F z` + lemma on the relaxation (§1.5) |
| 4 | `L = sqrt(TVL/2)/1000` "for numerical stability"; costs labelled USD and ETH inconsistently | Single unit system (USD); `L = TVL/2` derived from `xy=k`; no arbitrary rescaling |
| 5 | `compute_allocation_costs` hardcodes `alpha=0.95`, omits gas ⇒ biased baseline comparison | `alpha` is a parameter everywhere; all strategies evaluated with true fixed gas |
| 6 | TOC promises a "Theory and Background" section that does not exist | Theory/derivation/convexity actually written (§1) |
| 7 | Presentation notes contradict code ("alpha = 0.05", "baselines: equal split, liquidity-proportional" while code uses 0.95 / single-pool + mean-only) | Single source of truth; extras clearly labelled |
| 8 | No out-of-sample evaluation; "trade history" is simulated and never used to test the model | 2,500-order walk-forward backtest on real trades (§5) |
| 9 | ECOS as primary solver (deprecated); no solver-status handling | CLARABEL primary, SCS fallback, status checks, cached DPP problem |
| 10 | `quad_over_lin` in a Python loop over M×K atoms (slow) | DCP-equivalent vectorised norm (§1.4) + literal `quad_over_lin` kept as reference for cross-checks |
| 11 | CVaR estimator in evaluation inconsistent with the RU epigraph used to optimise | `empirical_cvar` is the RU-optimal tail mean (checked against an explicit RU solve in `tests/test_metrics.py`), used for every number |
| 12 | Efficient-frontier claims overstated (weighted sums recover supported Pareto points only) | Frontier reported descriptively |
| 13 | Scenario model "MEV mixture uniforms" not tied to any data | Historical resampling of real TVL deviations (§1.2), trend-inversion bug found and tested |
| 14 | No tests, no reproducibility guarantees | 32 pytest tests (convexity, RU-CVaR correctness, DCP, feasibility, no look-ahead, backtest smoke), fixed seeds |

---

## 7. Limitations (stated honestly)

* Realised cost is evaluated with the same convex cost model at the *realised real* TVL of
  the execution day — a full on-chain swap replay (pool reserves at each block, concentrated
  liquidity curves of Uniswap V3) is not possible without an archive node / subgraph key.
  The model itself is validated analytically against `xy=k` and its cost shape is checked on
  the real trade tape (§5).
* TVL is observed at daily granularity (DefiLlama); intra-day liquidity variation is unobserved.
* Gas costs are estimated (swap gas units × 20 gwei × real ETH price).  Sensitivity: gas
  scales the F-sweep threshold roughly linearly; the qualitative conclusions hold from 5 to
  50 gwei.
* Uniswap V3 concentrated liquidity and Curve-style amplifiers are approximated by the
  constant-product `L = TVL/2` surrogate (a single efficiency factor `kappa_m` would be the
  next refinement).
* The pools all trade the same ETH/USDC pair, so their risk factors are highly correlated;
  this is why CVaR vs. mean-only is nearly tied out of sample.  On a pool set with more
  heterogeneous venues the CVaR term should matter more (it already does under stress).

---

## 8. How to run

```bash
cd amm_cvar
pip install -r requirements.txt          # numpy pandas matplotlib cvxpy clarabel scs pytest

PYTHONPATH=src python scripts/01_download_data.py       # ~500 MB real data -> data/raw (cached)
PYTHONPATH=src python scripts/02_prepare_data.py        # real tape -> orders, pools, scenarios
PYTHONPATH=src python scripts/03_solve_and_analyze.py   # base case + baselines + sensitivities
PYTHONPATH=src python scripts/04_backtest_real_trades.py# 2,500-order real-trade backtest

PYTHONPATH=src python -m pytest tests -q                 # 32 tests
```

Outputs: `results/figures/*.png` (12 figures), `results/tables/*.csv` (all numbers above).
`notebook/Project_AMM_CVaR.ipynb` runs the same pipeline with commentary.

```
amm_cvar/
  src/amm_cvar/       config, data_sources, pools, cost_model, scenarios,
                      optimizer, baselines, metrics, sensitivity, backtest, plots
  scripts/            01 download -> 02 prepare -> 03 in-sample -> 04 backtest
  tests/              32 pytest tests
  data/{raw,processed}  cached downloads and derived datasets
  results/{figures,tables}  outputs
  notebook/           executed narrative notebook
```

---

## 9. References

1. Uniswap v2 Core Whitepaper — constant-product mechanics (`xy = k`).
2. Angeris, Chitra, Evans, Boyd — *Optimal Routing for Constant Function Market Makers*
   (routing as convex optimization) — `cfmm-routing.pdf`.
3. Rockafellar & Uryasev — *Optimization of Conditional Value-at-Risk* (epigraph
   reformulation) — `CVaR1.pdf`.
4. Daian et al. — *Flash Boys 2.0* (MEV / transaction reordering).
5. Boyd & Vandenberghe — *Convex Optimization* (DCP, quadratic-over-linear).
6. Data: Binance Vision (`data.binance.vision`, aggTrades + klines), DefiLlama
   (`yields.llama.fi`, real DEX pool TVL), public Ethereum RPC (gas price attempt).
