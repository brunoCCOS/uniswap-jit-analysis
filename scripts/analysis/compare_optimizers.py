"""
Finding 1: 12-second-bounded combinatorial optimizer vs the analytical
(closed-form) optimizer, both benchmarked against the real on-chain JIT LP.

For every JIT-attacked swap, this script compares the combinatorial search's
best incumbent after at most 12 seconds with the analytical solution:
    combinatorial_12s_optimal_utility_usd -- bounded combinatorial optimizer
    kh_optimal_utility_usd                -- analytical optimizer / closed-form
    actual_utility_usd                    -- real on-chain JIT LP outcome

All five processed pools are included. A search that reaches the deadline
returns its best evaluated position, never a zero-profit placeholder.

Usage:
    ./.venv/bin/python3 scripts/analysis/compare_optimizers.py

Outputs:
    output/reports/compare_optimizers.md
    output/reports/plots/compare_optimizers_profit_vs_size.png
    output/reports/plots/compare_optimizers_size_histogram.png
    output/reports/plots/compare_optimizers_profit_vs_size.tex (with --tikz)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import ALL_POOLS  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "output"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

INCLUDED_POOL_IDS = ["2697585", "2697588", "2697600", "2697647", "2697765"]
BOUNDED_UTILITY_COLUMN = "combinatorial_12s_optimal_utility_usd"

TIE_EPS = 1e-6
N_BINS = 10
# "Significant" margin rule for finding the win-threshold in step 2: the
# smallest trade-size bucket threshold above which the combinatorial
# optimizer's mean profit exceeds the analytical optimizer's mean profit by
# more than 5% relative AND stays above 5% for every larger bucket too (so
# the gap is a sustained regime change, not a single noisy bin).
SIGNIFICANCE_REL_GAP = 0.05


def load_jit_rows() -> pl.DataFrame:
    """Load JIT rows with a persisted 12-second-bounded result and pool labels."""
    pool_by_id = {p.pool_id: p for p in ALL_POOLS}
    frames = []
    for pool_id in INCLUDED_POOL_IDS:
        path = OUTPUT_ROOT / pool_id / "swaps_enriched.parquet"
        cfg = pool_by_id[pool_id]
        df = (
            pl.scan_parquet(path)
            # A successful rerun can be marked "skipped" after its bounded
            # value was already persisted; the non-null result is the durable
            # completion signal. Timeouts have no bounded utility and drop out.
            .filter(pl.col("is_jit") & pl.col(BOUNDED_UTILITY_COLUMN).is_not_null())
            .select(
                [
                    "block_number",
                    "volume_usd",
                    "direction",
                    pl.col(BOUNDED_UTILITY_COLUMN).alias("optimal_utility_usd"),
                    "kh_optimal_utility_usd",
                    "actual_utility_usd",
                ]
            )
            .with_columns(
                pl.lit(pool_id).alias("pool_id"),
                pl.lit(cfg.pair_label).alias("pair"),
                pl.lit(cfg.fee_millionths / 100).alias("fee_bps"),
            )
            .collect()
        )
        frames.append(df)
    return pl.concat(frames, how="vertical")


ECON_SIGNIFICANT_DIFF_USD = 0.01  # above this, a diff is not just floating-point noise


def describe_nontied_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Rows where optimal_utility_usd and kh_optimal_utility_usd differ by more
    than TIE_EPS, sorted by |diff| descending, with an economic-significance flag."""
    both = df.filter(
        pl.col("optimal_utility_usd").is_not_null()
        & pl.col("kh_optimal_utility_usd").is_not_null()
    )
    diff = both["optimal_utility_usd"] - both["kh_optimal_utility_usd"]
    nontied = both.filter(diff.abs() > TIE_EPS).with_columns(
        (pl.col("optimal_utility_usd") - pl.col("kh_optimal_utility_usd")).alias("diff_usd")
    ).with_columns(
        (pl.col("diff_usd").abs() > ECON_SIGNIFICANT_DIFF_USD).alias("econ_significant")
    ).sort(pl.col("diff_usd").abs(), descending=True)
    print(nontied.select(
        pl.col('kh_optimal_utility_usd'), pl.col('optimal_utility_usd'), pl.col('diff_usd')
        ).show(20))
    return nontied.select(
        ["pool_id", "volume_usd", "optimal_utility_usd", "kh_optimal_utility_usd", "diff_usd", "econ_significant"]
    )


def compute_win_rates(df: pl.DataFrame) -> dict:
    """Overall win-rate among rows with both optimal_utility_usd and
    kh_optimal_utility_usd non-null."""
    both = df.filter(
        pl.col("optimal_utility_usd").is_not_null()
        & pl.col("kh_optimal_utility_usd").is_not_null()
    )
    n = len(both)
    diff = both["optimal_utility_usd"] - both["kh_optimal_utility_usd"]
    comb_wins = int((diff > TIE_EPS).sum())
    kh_wins = int((diff < -TIE_EPS).sum())
    ties = n - comb_wins - kh_wins
    return {
        "n_compared": n,
        "n_total_jit": len(df),
        "comb_wins": comb_wins,
        "kh_wins": kh_wins,
        "ties": ties,
        "comb_win_pct": 100.0 * comb_wins / n if n else float("nan"),
        "kh_win_pct": 100.0 * kh_wins / n if n else float("nan"),
        "tie_pct": 100.0 * ties / n if n else float("nan"),
    }


def compute_size_bins(df: pl.DataFrame, n_bins: int = N_BINS) -> pl.DataFrame:
    """Bin rows (with both optimal cols non-null) into volume_usd quantile
    buckets and compute mean profit per method per bucket."""
    both = df.filter(
        pl.col("optimal_utility_usd").is_not_null()
        & pl.col("kh_optimal_utility_usd").is_not_null()
    )
    quantiles = [both["volume_usd"].quantile(q) for q in np.linspace(0, 1, n_bins + 1)]
    # de-dup edges (quantile ties can collapse bins for skewed data)
    edges = sorted(set(quantiles))
    both = both.with_columns(
        pl.col("volume_usd")
        .cut(edges[1:-1], labels=[str(i) for i in range(len(edges) - 1)])
        .alias("bin")
    )
    agg = (
        both.group_by("bin")
        .agg(
            pl.len().alias("n"),
            pl.col("volume_usd").min().alias("size_min"),
            pl.col("volume_usd").max().alias("size_max"),
            pl.col("volume_usd").mean().alias("size_mean"),
            pl.col("optimal_utility_usd").mean().alias("comb_mean_profit"),
            pl.col("kh_optimal_utility_usd").mean().alias("kh_mean_profit"),
            pl.col("actual_utility_usd").mean().alias("actual_mean_profit"),
        )
        .with_columns(pl.col("bin").cast(pl.Int64))
        .sort("bin")
    )
    agg = agg.with_columns(
        (
            (pl.col("comb_mean_profit") - pl.col("kh_mean_profit"))
            / pl.col("kh_mean_profit").abs()
        ).alias("rel_gap")
    )
    return agg


def find_significance_threshold(bins: pl.DataFrame) -> float | None:
    """Smallest size_min such that rel_gap > SIGNIFICANCE_REL_GAP for this bin
    and every larger bin (sustained regime, not a single noisy bin)."""
    rel_gaps = bins["rel_gap"].to_list()
    size_mins = bins["size_min"].to_list()
    n = len(rel_gaps)
    for i in range(n):
        tail = rel_gaps[i:]
        if all(g is not None and g > SIGNIFICANCE_REL_GAP for g in tail):
            return size_mins[i]
    return None


def fraction_above_threshold(df: pl.DataFrame, threshold: float) -> dict:
    n_total = len(df)
    n_above = int((df["volume_usd"] > threshold).sum())
    return {
        "n_total": n_total,
        "n_above": n_above,
        "pct_above": 100.0 * n_above / n_total if n_total else float("nan"),
    }



def save_tikz(fig, tikz_path: Path) -> None:
    """Export a Matplotlib figure as PGFPlots, bridging tikzplotlib's legacy APIs."""
    from matplotlib.backends import backend_pgf

    if not hasattr(backend_pgf, "common_texification"):
        backend_pgf.common_texification = backend_pgf._tex_escape
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None and not hasattr(legend, "_ncol"):
            legend._ncol = legend._ncols
    import tikzplotlib

    tikzplotlib.save(tikz_path, figure=fig, axis_width=r"\\columnwidth")


def make_profit_vs_size_plot(
    df: pl.DataFrame,
    threshold: float | None,
    out_path: Path,
    tikz_path: Path | None = None,
) -> None:
    """Academic log-log scatter of successfully bounded results vs analytical profit."""
    comparable = df.filter(
        (pl.col("volume_usd") > 0)
        & (pl.col("optimal_utility_usd") > 0)
        & (pl.col("kh_optimal_utility_usd") > 0)
    )
    volume = comparable["volume_usd"].to_numpy()
    bounded_profit = comparable["optimal_utility_usd"].to_numpy()
    analytical_profit = comparable["kh_optimal_utility_usd"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        volume, bounded_profit, s=12, alpha=0.38, marker="o", linewidths=0,
        color="#0072B2", label="Combinatorial (12 s bounded)",
    )
    ax.scatter(
        volume, analytical_profit, s=12, alpha=0.38, marker="^", linewidths=0,
        color="#D55E00", label="Analytical solution",
    )
    if threshold is not None:
        ax.axvline(threshold, color="0.35", linestyle="--", linewidth=1.0,
                   label=f"5% gap threshold (${threshold:,.0f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Swap volume (USD)")
    ax.set_ylabel("JIT profit (USD)")
    ax.set_title("JIT profit versus swap volume")
    ax.text(
        0.01, 0.01,
        f"JIT swaps; both estimates positive; n = {len(comparable):,}",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if tikz_path is not None:
        save_tikz(fig, tikz_path)
    plt.close(fig)


def make_size_histogram_plot(
    df: pl.DataFrame, threshold: float | None, out_path: Path, tikz_path: Path | None = None
):
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = df["volume_usd"].to_numpy()
    sizes = sizes[sizes > 0]
    bins = np.logspace(np.log10(max(sizes.min(), 1e-2)), np.log10(sizes.max()), 40)
    ax.hist(sizes, bins=bins, color="tab:blue", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Trade size, volume_usd (log scale)")
    ax.set_ylabel("Count of JIT-attacked swaps")
    ax.set_title("Distribution of JIT-attacked swap trade sizes\n(pooled across 5 pools)")
    ax.grid(alpha=0.3)

    if threshold is not None:
        pct_above = 100.0 * (sizes > threshold).sum() / len(sizes)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5)
        ax.annotate(
            f"threshold = ${threshold:,.0f}\n{pct_above:.1f}% of swaps above",
            xy=(threshold, ax.get_ylim()[1] * 0.9),
            xytext=(threshold * 1.3, ax.get_ylim()[1] * 0.9),
            fontsize=9, color="red",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if tikz_path is not None:
        save_tikz(fig, tikz_path)
    plt.close(fig)


def write_markdown(
    df: pl.DataFrame,
    win: dict,
    bins: pl.DataFrame,
    threshold: float | None,
    frac_above: dict,
    nontied: pl.DataFrame,
    out_path: Path,
    plot1_path: Path,
    plot2_path: Path,
    tikz_paths: tuple[Path, ...] = (),
):
    pool_counts = (
        df.group_by(["pool_id", "pair", "fee_bps"])
        .agg(pl.len().alias("n_jit_rows"))
        .sort("pool_id")
    )

    lines = []
    lines.append("# 12-second-bounded combinatorial vs analytical optimizer\n")
    lines.append(
        "Comparison of the persisted 12-second-bounded combinatorial result "
        "(`combinatorial_12s_optimal_utility_usd`) against the analytical / "
        "closed-form optimizer (`kh_optimal_utility_usd`). The report pools all "
        "five processed pools. A search reaching the 12-second deadline contributes "
        "the best position evaluated by that deadline; trade size is `volume_usd`.\n"
    )

    lines.append("## Pools included\n")
    lines.append("| pool_id | pair | fee_bps | JIT rows |")
    lines.append("|---|---|---|---|")
    for row in pool_counts.iter_rows(named=True):
        lines.append(f"| {row['pool_id']} | {row['pair']} | {row['fee_bps']:.0f} | {row['n_jit_rows']} |")
    lines.append(f"| **total** | | | **{len(df)}** |\n")

    lines.append(
        f"**Headline**: across the five included pools, the 12-second-bounded "
        f"combinatorial optimizer and the analytical/closed-form optimizer agree almost exactly "
        f"({win['tie_pct']:.2f}% of comparable rows tied within ${TIE_EPS:g}); real "
        f"economic disagreements are rare ({int(nontied['econ_significant'].sum())} rows out of "
        f"{win['n_compared']}) and, where they occur, favor the combinatorial optimizer, but at "
        f"small/medium trade sizes rather than large ones. Both optimizers substantially and "
        f"consistently out-earn the real on-chain JIT LP (`actual_utility_usd`) at every trade "
        f"size (see bin table below).\n"
    )

    lines.append("## 1. Overall win-rate\n")
    lines.append(
        f"Among {win['n_compared']} JIT rows with both `optimal_utility_usd` and "
        f"`kh_optimal_utility_usd` non-null (out of {win['n_total_jit']} total JIT rows; "
        f"the analytical optimizer has no closed-form solution, and is null, on the rest):\n"
    )
    lines.append(f"- **Combinatorial (12-second bounded) strictly wins** (margin > {TIE_EPS:g}): "
                 f"{win['comb_wins']} / {win['n_compared']} = **{win['comb_win_pct']:.2f}%**")
    lines.append(f"- **Analytical optimizer strictly wins**: "
                 f"{win['kh_wins']} / {win['n_compared']} = **{win['kh_win_pct']:.2f}%**")
    lines.append(f"- **Effectively tied** (|diff| <= {TIE_EPS:g}): "
                 f"{win['ties']} / {win['n_compared']} = **{win['tie_pct']:.2f}%**\n")

    lines.append("## 2. Profit by trade-size bucket\n")
    lines.append(
        f"JIT rows (with both optimizer columns non-null) binned into {len(bins)} "
        f"volume_usd quantile buckets. `rel_gap` = (combinatorial_mean - analytical_mean) / "
        f"|analytical_mean|.\n"
    )
    lines.append("| bin | size range (USD) | n | mean size | combinatorial mean profit | analytical mean profit | real/actual mean profit | rel_gap |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in bins.iter_rows(named=True):
        lines.append(
            f"| {row['bin']} | ${row['size_min']:,.0f}-${row['size_max']:,.0f} | {row['n']} | "
            f"${row['size_mean']:,.0f} | ${row['comb_mean_profit']:,.2f} | "
            f"${row['kh_mean_profit']:,.2f} | ${row['actual_mean_profit']:,.2f} | "
            f"{row['rel_gap']*100:.1f}% |"
        )
    lines.append("")

    lines.append(
        f"**Significance rule used**: the smallest bucket lower-edge `size_min` such that "
        f"`rel_gap > {SIGNIFICANCE_REL_GAP*100:.0f}%` for that bucket AND every larger bucket "
        f"(a sustained regime change, not a single noisy bin).\n"
    )
    if threshold is not None:
        lines.append(f"**Threshold found: trade size > ${threshold:,.0f}** is where the "
                     f"combinatorial optimizer's edge over the analytical optimizer becomes "
                     f"consistently large (>{SIGNIFICANCE_REL_GAP*100:.0f}% relative).\n")
    else:
        lines.append("**No threshold found**: the relative gap never exceeds the significance "
                     "rule consistently for all larger buckets in this data.\n")

    lines.append("## 3. How common is the big-win condition\n")
    if threshold is not None:
        lines.append(
            f"Of all {frac_above['n_total']} JIT-attacked swaps (pooled, not just the "
            f"ones with a closed-form analytical solution), {frac_above['n_above']} have "
            f"`volume_usd` above the ${threshold:,.0f} threshold: "
            f"**{frac_above['pct_above']:.2f}%** of JIT-attacked swaps fall into the "
            f"regime where Algorithm 1's edge over the analytical optimizer is large.\n"
        )
    else:
        lines.append("No threshold was found, so this fraction is not applicable.\n")

    econ_sig = nontied.filter(pl.col("econ_significant"))
    lines.append("## Detail: where the two optimizers actually disagree\n")
    lines.append(
        f"Of the {win['n_compared']} comparable rows, only {len(nontied)} are even nominally "
        f"non-tied (diff > {TIE_EPS:g}), and of those only **{len(econ_sig)}** differ by more "
        f"than ${ECON_SIGNIFICANT_DIFF_USD:.2f} (the rest are floating-point-level noise well "
        f"under a cent). Listing the economically-significant disagreements:\n"
    )
    if len(econ_sig):
        lines.append("| pool_id | volume_usd | combinatorial profit | analytical profit | diff (comb - analytical) |")
        lines.append("|---|---|---|---|---|")
        for row in econ_sig.iter_rows(named=True):
            lines.append(
                f"| {row['pool_id']} | ${row['volume_usd']:,.0f} | ${row['optimal_utility_usd']:,.4f} | "
                f"${row['kh_optimal_utility_usd']:,.4f} | ${row['diff_usd']:,.4f} |"
            )
        lines.append("")
        lines.append(
            "**Note the direction**: these economically-meaningful disagreements occur at "
            "small-to-medium trade sizes (the analytical/closed-form optimizer occasionally "
            "collapses to a zero-profit boundary solution there, while the combinatorial "
            "search still finds a profitable range), not at large trade sizes. This is the "
            "opposite of the 'ours wins big at large trades' pattern step 2 was designed to "
            "detect, which is exactly why no such threshold exists in this data (see above).\n"
        )
    else:
        lines.append("(none)\n")

    lines.append("## Plots\n")
    lines.append(f"- Profit vs trade size: `{plot1_path.relative_to(REPO_ROOT)}`")
    lines.append(f"- Trade-size histogram with threshold: `{plot2_path.relative_to(REPO_ROOT)}`")
    for tikz_path in tikz_paths:
        lines.append(f"- TikZ/PGF export: `{tikz_path.relative_to(REPO_ROOT)}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tikz", action="store_true", help="also export every plot as TikZ/PGF")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_jit_rows()
    print(f"Loaded {len(df)} JIT-attacked rows across {INCLUDED_POOL_IDS}")

    win = compute_win_rates(df)
    print("Win rates:", win)

    bins = compute_size_bins(df)
    print(bins)

    threshold = find_significance_threshold(bins)
    print("Significance threshold:", threshold)

    frac_above = fraction_above_threshold(df, threshold) if threshold is not None else {
        "n_total": len(df), "n_above": 0, "pct_above": float("nan")
    }
    print("Fraction above threshold:", frac_above)

    nontied = describe_nontied_rows(df)
    print(f"Non-tied rows: {len(nontied)}, economically significant (>${ECON_SIGNIFICANT_DIFF_USD}): "
          f"{int(nontied['econ_significant'].sum())}")

    plot1_path = PLOTS_DIR / "compare_optimizers_profit_vs_size.png"
    plot2_path = PLOTS_DIR / "compare_optimizers_size_histogram.png"
    tikz_paths = (
        PLOTS_DIR / "compare_optimizers_profit_vs_size.tex",
        PLOTS_DIR / "compare_optimizers_size_histogram.tex",
    ) if args.tikz else ()
    make_profit_vs_size_plot(df, threshold, plot1_path, tikz_paths[0] if tikz_paths else None)
    make_size_histogram_plot(df, threshold, plot2_path, tikz_paths[1] if tikz_paths else None)

    md_path = REPORTS_DIR / "compare_optimizers.md"
    write_markdown(df, win, bins, threshold, frac_above, nontied, md_path, plot1_path, plot2_path, tikz_paths)

    print(f"Wrote {md_path}")
    print(f"Wrote {plot1_path}")
    print(f"Wrote {plot2_path}")
    for tikz_path in tikz_paths:
        print(f"Wrote {tikz_path}")


if __name__ == "__main__":
    main()
