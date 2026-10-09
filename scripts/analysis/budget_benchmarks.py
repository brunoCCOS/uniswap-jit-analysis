"""
Finding: how much of the JIT opportunity is captured under two different
capital-availability benchmarks, since the real JIT LP's true available
capital is unobserved.

  (i)  SAME CAPITAL  -- `optimal_utility_usd`, already in swaps_enriched.parquet:
       the combinatorial optimizer (Algorithm 1 / "ours") run with
       budget = jit_liquidity_usd, i.e. exactly the USD capital the real
       on-chain JIT LP actually deployed. No new compute needed; read
       straight from the parquet for is_jit==True rows.

  (ii) FREE BUDGET -- a NEW optimizer re-run with budget = 1e12 USD
       (effectively unconstrained), using the same combinatorial optimizer
       bridge (src/optimizers.py -> JITUniswapOptimization's
       Utility.optimize(method="combinatorial")). This requires replaying
       the full per-pool event stream through src.enricher.enrich_pool so
       that PoolState (tick deltas, active liquidity) is correctly
       reconstructed at each JIT swap, then calling the combinatorial
       optimizer with the larger budget instead of jit_liquidity_usd. The
       analytical/closed-form optimizer is NOT re-run here (monkeypatched
       to a no-op) since this benchmark only concerns Algorithm 1's free-
       budget ceiling.

Because the free-budget pass is a new, slower compute pass (full event
replay per pool), it is restricted to the two SMALLER pools:
    2697585  USDC/WETH  fee=100bps  (9,127 events,   7 JIT swap-rows)
    2697647  WBTC/USDC  fee=30bps   (43,173 events, 169 JIT swap-rows)
2697600 and 2697765 are skipped for the free-budget re-run (compute cost:
2697600 has 627 JIT rows / 247k events, 2697765 has 9,037 JIT rows / 1.18M
events -- both would take substantially longer to replay and re-optimize
than the two small pools, and the combinatorial optimizer's per-swap cost
scales with the tick-span of each swap, not just event count). Same-capital
numbers for all four pools are reported for context since they require no
new compute.

Pool 2697588 (USDC/USDT, fee=5bps) is excluded, pending a separate fix for a
combinatorial-optimizer hang on an outlier-sized swap.

The free-budget pass is run on ALL is_jit==True rows in each of the two
pools (not a further subsample): 7 rows for 2697585, 169 rows for 2697647.
This is tractable because the combinatorial optimizer is only invoked on
JIT rows in the first place (see src/enricher.py:_handle_swap), and the
candidate-range count it searches is governed by the swap's no-JIT tick
span (comb_ts_mult), not by the budget -- so a free budget does not blow up
the per-swap search cost, only the per-candidate liquidity ceiling.

Usage:
    ./.venv/bin/python3 scripts/analysis/budget_benchmarks.py

Outputs:
    output/reports/budget_benchmarks.md
    output/reports/plots/budget_benchmarks_scatter.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import POOL_BY_ID  # noqa: E402
from src.loader import load_events  # noqa: E402
import src.enricher as enricher  # noqa: E402
import src.optimizers as optimizers  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "output"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

ALL_INCLUDED_POOL_IDS = ["2697585", "2697600", "2697647", "2697765"]
EXCLUDED_POOL_ID = "2697588"

# Free-budget re-run restricted to these two (smaller) pools -- see module
# docstring for why 2697600 / 2697765 are skipped.
FREE_BUDGET_POOL_IDS = ["2697585", "2697647"]
FREE_BUDGET_USD = 1e12  # effectively unconstrained

_orig_run_simulation_optimizer = optimizers.run_simulation_optimizer


def _free_budget_sim_optimizer(*args, **kwargs):
    """Wrap run_simulation_optimizer, overriding budget to FREE_BUDGET_USD."""
    kwargs = dict(kwargs)
    kwargs["budget"] = FREE_BUDGET_USD
    return _orig_run_simulation_optimizer(*args, **kwargs)


def _skip_analytical_optimizer(*args, **kwargs):
    """No-op stand-in for run_analytical_optimizer: this benchmark only
    concerns the combinatorial optimizer's free-budget ceiling, so skip the
    (separately costed) analytical re-run entirely."""
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_same_capital(pool_id: str) -> pl.DataFrame:
    """Same-capital benchmark: optimal_utility_usd already in the parquet,
    computed with budget=jit_liquidity_usd. No new compute."""
    path = OUTPUT_ROOT / pool_id / "swaps_enriched.parquet"
    return (
        pl.scan_parquet(path)
        .filter(pl.col("is_jit"))
        .select(
            [
                "block_number",
                "transaction_hash",
                "jit_liquidity_usd",
                "optimal_utility_usd",
                "actual_utility_usd",
            ]
        )
        .rename({"optimal_utility_usd": "same_capital_profit_usd"})
        .collect()
    )


def run_free_budget_pass(pool_id: str) -> pl.DataFrame:
    """Replay the full event stream for one pool through enrich_pool with
    the combinatorial optimizer's budget overridden to FREE_BUDGET_USD.
    Returns is_jit==True rows with the free-budget optimal_utility_usd."""
    cfg = POOL_BY_ID[pool_id]
    df = load_events(cfg)

    enricher.run_simulation_optimizer = _free_budget_sim_optimizer
    enricher.run_analytical_optimizer = _skip_analytical_optimizer
    try:
        swaps_df, _, _ = enricher.enrich_pool(df, cfg)
    finally:
        enricher.run_simulation_optimizer = _orig_run_simulation_optimizer
        enricher.run_analytical_optimizer = optimizers.run_analytical_optimizer

    return (
        swaps_df.filter(pl.col("is_jit"))
        .select(["block_number", "transaction_hash", "optimal_utility_usd"])
        .rename({"optimal_utility_usd": "free_budget_profit_usd"})
    )


def build_pool_comparison(pool_id: str) -> pl.DataFrame:
    """Join same-capital and free-budget results per JIT swap for one pool."""
    same_cap = load_same_capital(pool_id)
    free_budget = run_free_budget_pass(pool_id)
    joined = same_cap.join(
        free_budget, on=["block_number", "transaction_hash"], how="inner"
    )
    return joined.with_columns(pl.lit(pool_id).alias("pool_id"))


# ──────────────────────────────────────────────────────────────────────────────
# Summary stats
# ──────────────────────────────────────────────────────────────────────────────


def summarize(df: pl.DataFrame, same_col: str, free_col: str | None = None) -> dict:
    n = len(df)
    same = df[same_col]
    out = {
        "n": n,
        "same_capital_mean": same.mean() if n else float("nan"),
        "same_capital_median": same.median() if n else float("nan"),
    }
    if free_col is not None:
        free = df[free_col]
        out["free_budget_mean"] = free.mean() if n else float("nan")
        out["free_budget_median"] = free.median() if n else float("nan")
        out["ratio_mean"] = (
            out["free_budget_mean"] / out["same_capital_mean"]
            if out["same_capital_mean"]
            else float("nan")
        )
        out["ratio_median"] = (
            out["free_budget_median"] / out["same_capital_median"]
            if out["same_capital_median"]
            else float("nan")
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────


def make_scatter_plot(frames: dict[str, pl.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"2697585": "tab:blue", "2697647": "tab:orange"}
    pool_labels = {
        "2697585": "2697585 USDC/WETH fee=100bps",
        "2697647": "2697647 WBTC/USDC fee=30bps",
    }

    all_x, all_y = [], []
    for pool_id, df in frames.items():
        x = df["same_capital_profit_usd"].to_numpy()
        y = df["free_budget_profit_usd"].to_numpy()
        all_x.append(x)
        all_y.append(y)
        ax.scatter(
            x, y, s=28, alpha=0.7, color=colors[pool_id],
            label=f"{pool_labels[pool_id]} (n={len(df)})",
        )

    all_x = np.concatenate(all_x) if all_x else np.array([0.0])
    all_y = np.concatenate(all_y) if all_y else np.array([0.0])
    lo = min(all_x.min(), all_y.min(), 0.0)
    hi = max(all_x.max(), all_y.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x (no benefit from extra budget)")

    ax.set_xlabel("Same-capital profit, optimal_utility_usd @ budget=jit_liquidity_usd (USD)")
    ax.set_ylabel(f"Free-budget profit, optimal_utility_usd @ budget=${FREE_BUDGET_USD:.0e} (USD)")
    ax.set_title("Same-capital vs free-budget optimal JIT profit per swap\n(2697585 + 2697647, is_jit==True rows)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown
# ──────────────────────────────────────────────────────────────────────────────


def write_markdown(
    all_pool_same_capital: dict[str, dict],
    paired_summaries: dict[str, dict],
    plot_path: Path,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Budget benchmarks: same-capital vs free-budget JIT profit\n")
    lines.append(
        "The real on-chain JIT LP's true available capital is unobserved, so we "
        "bound Algorithm 1's (combinatorial optimizer's) profit under two "
        "benchmarks:\n"
    )
    lines.append(
        "- **SAME CAPITAL** (`optimal_utility_usd` as already stored in "
        "`swaps_enriched.parquet`): Algorithm 1 run with `budget = "
        "jit_liquidity_usd`, i.e. exactly the USD capital the real JIT LP "
        "actually deployed on-chain. No new compute.\n"
    )
    lines.append(
        f"- **FREE BUDGET** (new re-run, this script): Algorithm 1 re-run with "
        f"`budget = ${FREE_BUDGET_USD:.0e}` (effectively unconstrained), via the "
        "same `src/optimizers.py` bridge into `JITUniswapOptimization`'s "
        "`Utility.optimize(method=\"combinatorial\")`, after replaying each "
        "pool's full event stream so `PoolState` is correctly reconstructed at "
        "each JIT swap.\n"
    )
    lines.append(
        "Pool 2697588 (USDC/USDT, fee=5bps) is excluded, pending a separate fix "
        "for a combinatorial-optimizer hang on an outlier-sized swap.\n"
    )

    lines.append("## Compute-cost scope\n")
    lines.append(
        "The free-budget re-run replays the full event stream per pool (to "
        "reconstruct `PoolState` correctly at each JIT swap) and is restricted "
        "to the two smaller pools:\n"
    )
    lines.append("| pool_id | pair | fee_bps | events | JIT swap-rows re-run |")
    lines.append("|---|---|---|---|---|")
    for pool_id in FREE_BUDGET_POOL_IDS:
        cfg = POOL_BY_ID[pool_id]
        n = paired_summaries[pool_id]["n"]
        meta = json.loads((OUTPUT_ROOT / pool_id / "metadata.json").read_text())
        lines.append(
            f"| {pool_id} | {cfg.pair_label} | {cfg.fee_millionths/100:.0f} | "
            f"{meta['total_events']:,} | {n} (all JIT rows, no subsampling) |"
        )
    lines.append("")
    lines.append(
        "2697600 (627 JIT rows / 247k events) and 2697765 (9,037 JIT rows / "
        "1.18M events) are skipped for the free-budget re-run: compute cost. "
        "Their same-capital numbers (already computed, no new compute) are "
        "shown below for context.\n"
    )

    lines.append("## Same-capital profit, all four fresh pools (context, no new compute)\n")
    lines.append("| pool_id | pair | fee_bps | n JIT rows | mean profit (USD) | median profit (USD) |")
    lines.append("|---|---|---|---|---|---|")
    pool_meta = {
        "2697585": ("USDC/WETH", 100),
        "2697600": ("USDC/WETH", 30),
        "2697647": ("WBTC/USDC", 30),
        "2697765": ("USDC/WETH", 5),
    }
    for pool_id in ALL_INCLUDED_POOL_IDS:
        s = all_pool_same_capital[pool_id]
        pair, fee = pool_meta[pool_id]
        lines.append(
            f"| {pool_id} | {pair} | {fee} | {s['n']} | "
            f"${s['same_capital_mean']:,.2f} | ${s['same_capital_median']:,.2f} |"
        )
    lines.append("")

    lines.append("## Same-capital vs free-budget, paired per JIT swap (new compute, 2 pools)\n")
    lines.append(
        "| pool_id | pair | fee_bps | n | same-capital mean | same-capital median | "
        "free-budget mean | free-budget median | ratio (mean) | ratio (median) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for pool_id in FREE_BUDGET_POOL_IDS:
        s = paired_summaries[pool_id]
        pair, fee = pool_meta[pool_id]
        lines.append(
            f"| {pool_id} | {pair} | {fee} | {s['n']} | "
            f"${s['same_capital_mean']:,.2f} | ${s['same_capital_median']:,.2f} | "
            f"${s['free_budget_mean']:,.2f} | ${s['free_budget_median']:,.2f} | "
            f"{s['ratio_mean']:.2f}x | {s['ratio_median']:.2f}x |"
        )
    lines.append("")

    lines.append("## Interpretation\n")
    for pool_id in FREE_BUDGET_POOL_IDS:
        s = paired_summaries[pool_id]
        pair, fee = pool_meta[pool_id]
        lines.append(
            f"- **{pool_id} ({pair}, {fee}bps)**: with the capital the real JIT LP "
            f"actually deployed, Algorithm 1 captures a mean profit of "
            f"${s['same_capital_mean']:,.2f} per JIT swap (median "
            f"${s['same_capital_median']:,.2f}). With an unconstrained budget, the "
            f"best-case profit rises to a mean of ${s['free_budget_mean']:,.2f} "
            f"(median ${s['free_budget_median']:,.2f}) -- "
            f"**{s['ratio_mean']:.2f}x** the same-capital mean "
            f"({s['ratio_median']:.2f}x at the median). In other words, the "
            f"same-capital constraint captures roughly "
            f"{100/s['ratio_mean']:.1f}% (mean) / {100/s['ratio_median']:.1f}% "
            f"(median) of the best-case (unconstrained-budget) opportunity "
            f"value for this pool.\n"
        )

    lines.append("## Plot\n")
    lines.append(f"- Same-capital vs free-budget profit scatter: `{plot_path.relative_to(REPO_ROOT)}`\n")

    out_path.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading same-capital (already computed) profit for all four fresh pools...")
    all_pool_same_capital = {}
    for pool_id in ALL_INCLUDED_POOL_IDS:
        df = load_same_capital(pool_id)
        all_pool_same_capital[pool_id] = summarize(df, "same_capital_profit_usd")
        print(f"  {pool_id}: n={len(df)}, mean=${all_pool_same_capital[pool_id]['same_capital_mean']:,.2f}")

    print(f"\nRunning free-budget (budget=${FREE_BUDGET_USD:.0e}) re-run for {FREE_BUDGET_POOL_IDS}...")
    paired_frames: dict[str, pl.DataFrame] = {}
    paired_summaries: dict[str, dict] = {}
    for pool_id in FREE_BUDGET_POOL_IDS:
        print(f"  [{pool_id}] replaying event stream + re-optimizing JIT swaps...")
        joined = build_pool_comparison(pool_id)
        paired_frames[pool_id] = joined
        paired_summaries[pool_id] = summarize(
            joined, "same_capital_profit_usd", "free_budget_profit_usd"
        )
        s = paired_summaries[pool_id]
        print(
            f"  [{pool_id}] n={s['n']} same_cap_mean=${s['same_capital_mean']:,.2f} "
            f"free_budget_mean=${s['free_budget_mean']:,.2f} ratio={s['ratio_mean']:.2f}x"
        )

    plot_path = PLOTS_DIR / "budget_benchmarks_scatter.png"
    make_scatter_plot(paired_frames, plot_path)
    print(f"Wrote {plot_path}")

    md_path = REPORTS_DIR / "budget_benchmarks.md"
    write_markdown(all_pool_same_capital, paired_summaries, plot_path, md_path)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
