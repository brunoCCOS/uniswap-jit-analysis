"""
Lambda profile: where does a JIT position sit between the pre-swap price and
the no-JIT counterfactual final price?

For each JIT swap we compute a normalized log-price position

    lambda = (ln(q) - ln(q_f)) / (ln(q) - ln(q'))

where
    q  = initial_price        (price immediately before the swap)
    q_f = no_jit_final_price  (counterfactual final price with NO JIT at all;
                                the full price move "available" to the swap)
    q' = the price at the FAR tick boundary of the JIT position, i.e. the
         boundary on the same side the price is actually moving toward.

lambda ~ 0   -> position sits right at the pre-swap price (never really
                engages the move)
lambda ~ 1   -> position's far edge reaches exactly the no-JIT counterfactual
                price (perfectly "sized" to the move)
lambda > 1   -> position's far edge sits past the counterfactual price
                (overshoots / wider than the move)

We compute this twice per JIT swap:
  (a) the REAL on-chain JIT position   (jit_tick_lower / jit_tick_upper)
  (b) Algorithm 1's optimal position    (optimal_tick_lower / optimal_tick_upper)

Direction handling
-------------------
The parquet's `direction` column takes values "buy"/"sell" (tick-increasing /
tick-decreasing), not "up"/"down" of the human-readable price column used for
q and q_f. Because this repo's price convention (see src/price.py,
sqrt_x96_to_price: human price = token0 per token1) is *inversely* related to
tick for these pools (price decreases as tick increases), picking the "far"
boundary from the raw direction string would silently pick the wrong side.
Instead we derive the far boundary directly and robustly from the sign of the
actual price move (q_f vs q): whichever tick boundary's price lies on the same
side as q_f is the "far" boundary the price is moving toward. Since
sqrt_x96_to_price is monotonically decreasing in tick, price(tick_upper) <=
price(tick_lower) always, so:
    q' = price(tick_upper)  if q_f < q   (price falling)
    q' = price(tick_lower)  if q_f > q   (price rising)

Output:
    output/reports/lambda_profile.md
    output/reports/plots/lambda_vs_tradesize.png
    output/reports/plots/lambda_vs_budget.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import POOL_BY_ID  # noqa: E402
from src.price import sqrt_x96_to_price, tick_to_sqrt_x96  # noqa: E402

POOL_IDS = ["2697585", "2697600", "2697647", "2697765"]
EXCLUDED_NOTE = (
    "Pool 2697588 (USDC/USDT, fee=5bps) is excluded: pending a separate fix for a "
    "combinatorial-optimizer hang on an outlier-sized swap, and its on-disk output is stale."
)

OUT_DIR = REPO_ROOT / "output" / "reports"
PLOT_DIR = OUT_DIR / "plots"


def tick_to_price(tick: int, dec0: int, dec1: int) -> float:
    """Reuse src/price.py conventions: tick -> sqrtPriceX96 -> human price."""
    return sqrt_x96_to_price(tick_to_sqrt_x96(int(tick)), dec0, dec1)


def load_jit_rows() -> pl.DataFrame:
    """Lazily scan each pool's parquet, keep only is_jit rows + needed columns."""
    frames = []
    cols = [
        "block_number",
        "transaction_hash",
        "direction",
        "volume_usd",
        "initial_price",
        "no_jit_final_price",
        "jit_tick_lower",
        "jit_tick_upper",
        "jit_liquidity_usd",
        "optimal_tick_lower",
        "optimal_tick_upper",
    ]
    for pool_id in POOL_IDS:
        parquet_path = REPO_ROOT / "output" / pool_id / "swaps_enriched.parquet"
        lf = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("is_jit"))
            .select(cols)
            .with_columns(pl.lit(pool_id).alias("pool_id"))
        )
        frames.append(lf.collect())
    return pl.concat(frames, how="vertical")


def compute_lambda(df: pl.DataFrame) -> pl.DataFrame:
    """Add lambda_real / lambda_optimal columns, rowwise (small post-filter N)."""
    dec_cache = {
        pool_id: (POOL_BY_ID[pool_id].token0_decimals, POOL_BY_ID[pool_id].token1_decimals)
        for pool_id in POOL_IDS
    }

    records = []
    for row in df.iter_rows(named=True):
        dec0, dec1 = dec_cache[row["pool_id"]]
        q = row["initial_price"]
        q_f = row["no_jit_final_price"]
        if q is None or q_f is None or q <= 0 or q_f <= 0 or q == q_f:
            continue  # drop no-price-movement / invalid rows

        price_falling = q_f < q

        def far_price(tick_lower, tick_upper):
            p_lower = tick_to_price(tick_lower, dec0, dec1)
            p_upper = tick_to_price(tick_upper, dec0, dec1)
            return p_upper if price_falling else p_lower

        q_real = far_price(row["jit_tick_lower"], row["jit_tick_upper"])
        q_opt = far_price(row["optimal_tick_lower"], row["optimal_tick_upper"])

        log_q = np.log(q)
        log_qf = np.log(q_f)
        numer = log_q - log_qf

        lam_real = np.nan if q_real <= 0 else numer / (log_q - np.log(q_real))
        lam_opt = np.nan if q_opt <= 0 else numer / (log_q - np.log(q_opt))

        records.append(
            {
                "pool_id": row["pool_id"],
                "volume_usd": row["volume_usd"],
                "jit_liquidity_usd": row["jit_liquidity_usd"],
                "lambda_real": lam_real,
                "lambda_optimal": lam_opt,
            }
        )

    out = pl.DataFrame(records)
    # drop infs (q' coincidentally == q) and non-finite
    out = out.filter(
        pl.col("lambda_real").is_finite() & pl.col("lambda_optimal").is_finite()
    )
    return out


def binned_series(x: np.ndarray, y: np.ndarray, n_bins: int = 12):
    """Log-spaced bins of x; return bin-center x, mean y, sem y per bin (bins with >=5 pts)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    log_x = np.log10(x)
    edges = np.linspace(log_x.min(), log_x.max(), n_bins + 1)
    centers, means, sems = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (log_x >= lo) & (log_x <= hi)
        else:
            sel = (log_x >= lo) & (log_x < hi)
        if sel.sum() < 5:
            continue
        centers.append(10 ** ((lo + hi) / 2))
        means.append(y[sel].mean())
        sems.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else 0.0)
    return np.array(centers), np.array(means), np.array(sems)


def make_plot(df: pl.DataFrame, x_col: str, title: str, xlabel: str, out_path: Path):
    x = df[x_col].to_numpy()
    y_real = df["lambda_real"].to_numpy()
    y_opt = df["lambda_optimal"].to_numpy()

    cx_real, cy_real, ce_real = binned_series(x, y_real)
    cx_opt, cy_opt, ce_opt = binned_series(x, y_opt)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        cx_real, cy_real, yerr=ce_real, marker="o", capsize=3, label="Real on-chain JIT (observed)",
        color="tab:red",
    )
    ax.errorbar(
        cx_opt, cy_opt, yerr=ce_opt, marker="s", capsize=3, label="Algorithm 1 / combinatorial (optimal)",
        color="tab:blue",
    )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="lambda=1 (reaches counterfactual price)")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("lambda  (0 = at pre-swap price, 1 = at no-JIT final price)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(df: pl.DataFrame, out_path: Path):
    lam_real = df["lambda_real"].to_numpy()
    lam_opt = df["lambda_optimal"].to_numpy()

    def stats(arr):
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "n": len(arr),
        }

    s_real = stats(lam_real)
    s_opt = stats(lam_opt)

    pool_labels = {
        "2697585": "2697585 USDC/WETH fee=100bps",
        "2697600": "2697600 USDC/WETH fee=30bps",
        "2697647": "2697647 WBTC/USDC fee=30bps",
        "2697765": "2697765 USDC/WETH fee=5bps",
    }
    per_pool_counts = df.group_by("pool_id").agg(pl.len().alias("n")).sort("pool_id")

    closer_to = lambda m: "the pre-swap price" if m < 0.5 else "the no-JIT counterfactual final price"

    lines = []
    lines.append("# Lambda profile: JIT position placement relative to the price move")
    lines.append("")
    lines.append(
        "Normalized log-price position lambda = (ln q - ln q_f) / (ln q - ln q'), where "
        "q = initial_price, q_f = no_jit_final_price (counterfactual price with no JIT at all), "
        "and q' = the price at the far tick boundary of the JIT position (the boundary on the side "
        "the price is actually moving toward, derived from the sign of q_f - q; see script docstring "
        "for why this must be derived from price rather than from the raw buy/sell direction label)."
    )
    lines.append("")
    lines.append(f"Pools included: {', '.join(pool_labels.values())}.")
    lines.append(f"({EXCLUDED_NOTE})")
    lines.append("")
    lines.append(f"Total JIT swaps analyzed (pooled, after dropping q==q_f and non-finite lambda): **{s_real['n']}**")
    lines.append("")
    lines.append("Per-pool JIT swap counts included in this analysis:")
    lines.append("")
    lines.append("| pool_id | n (is_jit rows w/ valid lambda) |")
    lines.append("|---|---|")
    for row in per_pool_counts.iter_rows(named=True):
        lines.append(f"| {row['pool_id']} | {row['n']} |")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| position | median lambda | mean lambda | n |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Real on-chain JIT LP (observed) | {s_real['median']:.3f} | {s_real['mean']:.3f} | {s_real['n']} |")
    lines.append(f"| Algorithm 1 / combinatorial (optimal) | {s_opt['median']:.3f} | {s_opt['mean']:.3f} | {s_opt['n']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"The real on-chain JIT LP has a median lambda of {s_real['median']:.3f} (mean {s_real['mean']:.3f}), "
        f"meaning its position's far boundary sits closer to {closer_to(s_real['median'])} on a log-price basis. "
        f"Algorithm 1's optimal position has a median lambda of {s_opt['median']:.3f} (mean {s_opt['mean']:.3f}), "
        f"sitting closer to {closer_to(s_opt['median'])}. "
    )
    if s_opt["median"] > s_real["median"]:
        lines.append(
            "Algorithm 1 places a systematically wider / further-reaching position than the real "
            "on-chain JIT LP for the same capital budget: the optimizer is willing to let its range span "
            "closer to (or past) the full no-JIT price move, while the observed on-chain LP tends to clamp "
            "its range tighter around the pre-swap price, capturing less of the available move. "
        )
    elif s_opt["median"] < s_real["median"]:
        lines.append(
            "Algorithm 1 places a systematically tighter position (closer to the pre-swap price) than the "
            "real on-chain JIT LP for the same capital budget. "
        )
    else:
        lines.append("The two strategies sit at a similar normalized distance from the pre-swap price on average. ")
    lines.append(
        "See the binned plots below for how lambda varies with trade size and with capital budget; if the two "
        "series' gap widens or narrows across the x-axis, the real/optimal divergence depends on swap size or budget "
        "rather than being a constant offset."
    )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![lambda vs trade size](plots/lambda_vs_tradesize.png)")
    lines.append("")
    lines.append("![lambda vs budget](plots/lambda_vs_budget.png)")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_jit_rows()
    df = compute_lambda(raw)

    assert len(df) > 0, "no valid JIT rows survived filtering"

    make_plot(
        df,
        "volume_usd",
        "Lambda vs trade size (binned means, pooled across 4 pools)",
        "trade size, volume_usd (log scale)",
        PLOT_DIR / "lambda_vs_tradesize.png",
    )
    make_plot(
        df,
        "jit_liquidity_usd",
        "Lambda vs capital budget (binned means, pooled across 4 pools)",
        "capital budget B, jit_liquidity_usd (log scale)",
        PLOT_DIR / "lambda_vs_budget.png",
    )
    write_markdown(df, OUT_DIR / "lambda_profile.md")

    print("Rows analyzed:", len(df))
    print("Real    median/mean lambda:", df["lambda_real"].median(), df["lambda_real"].mean())
    print("Optimal median/mean lambda:", df["lambda_optimal"].median(), df["lambda_optimal"].mean())


if __name__ == "__main__":
    main()
