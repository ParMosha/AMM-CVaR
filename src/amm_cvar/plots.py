"""Publication-style figures (saved to results/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

FIG = C.FIGURES_DIR

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------
def fig_data_overview(tvl: pd.DataFrame, orders: pd.DataFrame, deltas_hist: np.ndarray, pool_names: list[str]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for name in tvl.columns:
        ax.plot(tvl.index, tvl[name] / 1e6, label=name, lw=1.4)
    ax.set_title("Real pool TVL (DefiLlama)")
    ax.set_ylabel("TVL (USD millions)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 1]
    ax.hist(orders["F"], bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xscale("log")
    ax.set_title(f"Real parent orders (n={len(orders)}, 1-min taker sells)")
    ax.set_xlabel("Order size F (USD, log scale)")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.hist(deltas_hist, bins=40, color="coral", edgecolor="black", alpha=0.8)
    ax.set_title("Historical liquidity degradation $\\delta_{m,k}$ (real TVL drawdowns)")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("Count")

    ax = axes[1, 1]
    means = tvl.mean() / 1e6
    ax.barh(range(len(means)), means.values, color=plt.cm.Set2(np.linspace(0, 1, len(means))))
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=8)
    ax.set_title("Mean real TVL by pool")
    ax.set_xlabel("USD millions")

    return _save(fig, "01_data_overview.png")


def fig_impact_fit(impact_df: pd.DataFrame, fit: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bm = fit.get("bin_means") or {}
    if bm:
        yerr = np.maximum(bm.get("se_cost", np.zeros(len(bm["mean_notional"]))), 1e-6)
        ax.errorbar(
            bm["mean_notional"], bm["mean_cost"], yerr=yerr,
            fmt="o", color="steelblue", label="bin means ± s.e. (real tape)", zorder=3,
        )
    n = np.linspace(2e3, max(bm.get("mean_notional", [1e5])) * 1.05, 200)
    y = fit["a_tape"] * n + fit["b_tape"] * n**2
    ax.plot(n, y, "r-", lw=2, label=f"LSQ: {fit['a_tape']:.2e} n + {fit['b_tape']:.2e} n²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clip notional n (USD)")
    ax.set_ylabel("Realised impact cost (USD)")
    ax.set_title(f"Execution cost vs size on real tape (R²={fit['r2']:.2f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    if bm:
        a_only = fit["a_tape"] * np.array(bm["mean_notional"])
        ax.plot(bm["mean_notional"], np.array(bm["mean_cost"]) - a_only, "o-", label="bin mean − linear fit")
    resid_null = impact_df["cost_usd"] - fit["a_tape"] * impact_df["notional"] - fit["b_tape"] * impact_df["notional"] ** 2
    ax.hist(resid_null.clip(-50, 200), bins=60, color="coral", edgecolor="black", alpha=0.7, label="raw residuals")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Residuals / convex part")
    ax.set_xlabel("USD")
    ax.legend(fontsize=8)

    return _save(fig, "02_impact_fidelity.png")


# ---------------------------------------------------------------------------
# Base case + baselines
# ---------------------------------------------------------------------------
def fig_base_case(allocation: np.ndarray, names: list[str], F: float, scen_costs: dict[str, np.ndarray], alpha: float) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax.bar(range(len(names)), allocation, color=colors, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Allocation f_m (USD)")
    ax.set_title(f"CVaR-optimal split (F=${F:,.0f})")

    ax = axes[1]
    for (label, costs), color in zip(scen_costs.items(), ["C0", "C3", "C2"]):
        ax.hist(costs, bins=30, alpha=0.45, label=label, color=color, edgecolor="black")
    ax.set_xlabel("Scenario cost (USD)")
    ax.set_ylabel("Count")
    ax.set_title(f"Scenario cost distributions (α={alpha})")
    ax.legend(fontsize=7)

    ax = axes[2]
    fracs = allocation / allocation.sum()
    ax.pie(fracs, labels=[n.replace(" ", "\n") for n in names], textprops={"fontsize": 7})
    ax.set_title("Allocation shares")

    return _save(fig, "03_base_case_allocation.png")


def fig_baselines(summary: pd.DataFrame, scen_costs: dict[str, np.ndarray], alpha: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    cols = ["mean_cost", f"CVaR_{alpha:.2f}", "max_cost"]
    cols = [c for c in cols if c in summary.columns]
    sub = summary[cols]
    x = np.arange(len(sub))
    w = 0.25
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1) * w, sub[col].values, w, label=col, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Cost (USD)")
    ax.set_title("In-sample scenario costs by strategy")
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, costs in scen_costs.items():
        sorted_c = np.sort(costs)
        ax.plot(sorted_c, np.linspace(0, 1, len(sorted_c)), lw=1.8, label=label)
    ax.set_xlabel("Scenario cost (USD)")
    ax.set_ylabel("ECDF")
    ax.set_title("Cost CDFs")
    ax.legend(fontsize=7)

    return _save(fig, "04_baselines_insample.png")


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def fig_sensitivity_alpha(df: pd.DataFrame, names: list[str], F: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(df["alpha"], df["mean_cost"], "s--", label="mean cost", color="C0")
    ax.plot(df["alpha"], df["CVaR_eval"], "o-", label="CVaR (eval@95%)", color="C3")
    ax.plot(df["alpha"], df["CVaR_own_alpha"], "^:", label="CVaR (own α)", color="C1")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Cost (USD)")
    ax.set_title(r"Sensitivity to $\alpha$ (risk aversion)")
    ax.legend(fontsize=8)

    ax = axes[1]
    allocs = np.vstack(df["allocation"].values) / F
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for m, name in enumerate(names):
        ax.bar(x, allocs[:, m], bottom=bottom, label=name, edgecolor="black", width=0.6)
        bottom += allocs[:, m]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.2f}" for a in df["alpha"]])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Allocation share")
    ax.set_title("Allocation vs risk aversion")
    ax.legend(fontsize=7, loc="lower right")

    return _save(fig, "05_sensitivity_alpha.png")


def fig_sensitivity_size(df: pd.DataFrame, names: list[str]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    ax.plot(df["F"], df["mean_cost_per_usd_bps"], "o-", label="CVaR router")
    ax.plot(df["F"], 1e4 * df["single_pool_mean"] / df["F"], "s--", label="single-pool best")
    ax.set_xscale("log")
    ax.set_xlabel("Order size F (USD)")
    ax.set_ylabel("Mean cost (bps of F)")
    ax.set_title("Cost per dollar vs order size")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(df["F"], df["n_active"], "o-", color="teal")
    ax.set_xscale("log")
    ax.set_xlabel("Order size F (USD)")
    ax.set_ylabel("# active pools (>1% F)")
    ax.set_title("When does splitting become necessary?")

    ax = axes[2]
    allocs = np.vstack(df["allocation_frac"].values)
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for m, name in enumerate(names):
        ax.bar(x, allocs[:, m], bottom=bottom, label=name, edgecolor="black", width=0.6)
        bottom += allocs[:, m]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f/1e3:.0f}k" for f in df["F"]], rotation=45)
    ax.set_xlabel("F (USD)")
    ax.set_ylabel("Allocation share")
    ax.set_title("Allocation shares vs order size")
    ax.legend(fontsize=7)

    return _save(fig, "06_sensitivity_size.png")


def fig_sensitivity_delta(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    for strat, grp in df.groupby("strategy"):
        ax.plot(grp["delta_scale"], grp["mean_cost"], "o--", label=f"{strat} mean")
        ax.plot(grp["delta_scale"], grp["CVaR"], "s-", label=f"{strat} CVaR")
    ax.set_xlabel(r"uncertainty scale ($\delta$ intensity)")
    ax.set_ylabel("Scenario cost (USD)")
    ax.set_title("How much does CVaR protect the plan?")
    ax.legend(fontsize=7)

    ax = axes[1]
    for strat, grp in df.groupby("strategy"):
        ax.plot(grp["delta_scale"], grp["max_cost"], "o-", label=strat)
    ax.set_xlabel(r"uncertainty scale")
    ax.set_ylabel("Worst scenario cost (USD)")
    ax.set_title("Worst-case cost vs uncertainty intensity")
    ax.legend(fontsize=8)

    return _save(fig, "07_sensitivity_delta.png")


def fig_frontier(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    sc = ax.scatter(df["mean_cost"], df["CVaR"], c=df["lam"], cmap="coolwarm", s=70, edgecolor="black")
    ax.plot(df["mean_cost"], df["CVaR"], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Mean cost (USD)")
    ax.set_ylabel("CVaR (USD)")
    ax.set_title("Efficient frontier (mean vs CVaR)")
    plt.colorbar(sc, ax=ax, label=r"$\lambda$")

    ax = axes[1]
    allocs = np.vstack(df["allocation"].values)
    for m in range(allocs.shape[1]):
        ax.plot(df["lam"], allocs[:, m], "o-", lw=1.5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Allocation (USD)")
    ax.set_title("Allocation vs risk weight")
    return _save(fig, "08_frontier.png")


def fig_scalability(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for M_val, grp in df.groupby("M"):
        axes[0].plot(grp["K"], grp["solve_time_ms"], "o-", label=f"M={M_val} (warm)")
        if "cold_time_ms" in grp:
            axes[0].plot(grp["K"], grp["cold_time_ms"], "s--", alpha=0.5, label=f"M={M_val} (cold)")
    axes[0].set_xlabel("K (scenarios)")
    axes[0].set_ylabel("Solve time (ms)")
    axes[0].set_title("Solver scalability (warm re-solve vs cold compile+solve)")
    axes[0].legend(fontsize=7)

    axes[1].scatter(df["n_variables"], df["solve_time_ms"], c=df["M"], cmap="viridis", s=80, edgecolor="black")
    axes[1].set_xlabel("# variables")
    axes[1].set_ylabel("Solve time (ms, warm)")
    axes[1].set_title("Solve time vs problem size")
    return _save(fig, "09_scalability.png")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def fig_backtest(results: pd.DataFrame, alpha: float) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for strat, grp in results.groupby("strategy"):
        s = np.sort(grp["realized_cost"].to_numpy())
        ax.plot(np.maximum(s, 1e-2), np.linspace(0, 1, len(s)), lw=1.6, label=strat)
    ax.set_xscale("log")
    ax.set_xlabel("realised cost (USD, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Realised execution cost on real trades (n={results['ts'].nunique()})")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    data = [grp["cost_bps"].to_numpy() for _, grp in results.groupby("strategy")]
    labels = [s for s, _ in results.groupby("strategy")]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("realised cost (bps of F, log)")
    ax.set_title("Cost in bps by strategy")
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    ax = axes[1, 0]
    for strat, grp in results.groupby("strategy"):
        cum = grp.sort_values("ts").groupby("ts")["realized_cost"].sum().cumsum()
        ax.plot(cum.index, cum.values, lw=1.6, label=strat)
    ax.set_yscale("log")
    ax.set_ylabel("cumulative realised cost (USD, log)")
    ax.set_title("Cumulative cost over the real tape")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=20, labelsize=7)

    ax = axes[1, 1]
    pivot = results.pivot_table(index="ts", columns="strategy", values="realized_cost")
    savings = pivot["SinglePoolMean"] - pivot["CVaR"]
    ax.hist(savings.clip(-2_000, 20_000), bins=60, color="seagreen", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(savings.mean(), color="darkred", lw=2, label=f"mean saving ${savings.mean():,.0f}")
    ax.set_xlabel("single-pool cost − CVaR cost (USD, clipped at ±20k)")
    ax.set_ylabel("count")
    ax.set_title("Per-order savings of CVaR routing vs single-pool")
    ax.legend(fontsize=8)

    return _save(fig, "10_backtest_overall.png")


def fig_backtest_tail(results: pd.DataFrame, alpha: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for strat, grp in results.groupby("strategy"):
        s = np.sort(grp["realized_cost"].to_numpy())
        tail = np.maximum(s[int((1 - 0.05) * len(s)):], 1e-2)
        ax.plot(np.linspace(0.95, 1.0, len(tail)), tail, "o-", label=strat, lw=1.6, ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("quantile")
    ax.set_ylabel("realised cost (USD, log)")
    ax.set_title("Tail of realised cost (worst 5%)")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels, cvars = [], []
    from .metrics import empirical_cvar

    for strat, grp in results.groupby("strategy"):
        labels.append(strat)
        cvars.append(empirical_cvar(grp["realized_cost"].to_numpy(), alpha))
    ax.bar(range(len(labels)), cvars, color="coral", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"realised CVaR$_{{{alpha}}}$ (USD, log)")
    ax.set_title(f"Realised CVaR$_{{{alpha}}}$ on real trades")

    return _save(fig, "11_backtest_tail.png")


def fig_backtest_by_size(by_size: pd.DataFrame, alpha: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    piv_mean = by_size.pivot(index="size_bucket", columns="strategy", values="mean")
    piv_cvar = by_size.pivot(index="size_bucket", columns="strategy", values=f"CVaR_{alpha:.2f}")
    piv_mean.plot.bar(ax=axes[0], edgecolor="black")
    axes[0].set_title("Mean realised cost by order size")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=7)
    piv_cvar.plot.bar(ax=axes[1], edgecolor="black")
    axes[1].set_title(f"Realised CVaR$_{{{alpha}}}$ by order size")
    axes[1].set_ylabel("USD")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=7)
    return _save(fig, "12_backtest_by_size.png")
