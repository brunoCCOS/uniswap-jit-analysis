"""
Tick-crossing distribution analysis.

Reads processed swap parquet files and produces a CSV showing how many
transactions cross 0, 1, 2, … N tick boundaries, broken down by pool and
split by swap direction and JIT status.

Usage:
    python -m scripts.analyze_tick_crossings
    python -m scripts.analyze_tick_crossings --pool 2697600
    python -m scripts.analyze_tick_crossings --no-jit
    python -m scripts.analyze_tick_crossings --out output/analysis/ticks.csv

--no-jit mode:
    For JIT swaps, replaces ticks_crossed with a counterfactual estimate:
        no_jit_ticks_crossed = ticks_crossed + floor(|no_jit_final_tick - final_tick| / tick_spacing)
    This adds the extra tick-spacings the swap would have traversed without JIT depth.
    For non-JIT swaps, ticks_crossed is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ALL_POOLS, POOL_BY_ID, PoolConfig

OUTPUT_ROOT = Path(__file__).parent.parent / "output"


def _load_swaps(cfg: PoolConfig, no_jit: bool = False) -> pl.DataFrame | None:
    path = OUTPUT_ROOT / cfg.pool_id / "swaps_enriched.parquet"
    if not path.exists():
        click.echo(f"[{cfg.pool_id}] No processed data at {path}, skipping.", err=True)
        return None
    df = pl.read_parquet(path).with_columns(
        pl.col("active_liq_start").cast(pl.Float64),
        pl.col("active_liq_end").cast(pl.Float64),
    )
    if no_jit:
        # For JIT swaps with a valid counterfactual tick, estimate additional crossings
        # in the extended range using tick_spacing as the unit.
        ts = cfg.tick_spacing
        df = df.with_columns(
            pl.when(pl.col("is_jit") & pl.col("no_jit_final_tick").is_not_null())
            .then(
                pl.col("ticks_crossed")
                + (
                    (pl.col("no_jit_final_tick") - pl.col("final_tick")).abs() // ts
                ).cast(pl.Int64)
            )
            .otherwise(pl.col("ticks_crossed"))
            .alias("ticks_crossed")
        )
    return df.with_columns(
        pl.lit(cfg.pool_id).alias("pool_id"),
        pl.lit(cfg.pair_label).alias("pair"),
        pl.lit(round(cfg.fee_millionths / 100, 2)).alias("fee_bps"),
    )


def compute_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per (pool_id, pair, fee_bps, ticks_crossed): swap count, volume, fees.
    Also includes cumulative count/pct columns.
    """
    agg = (
        df.group_by(["pool_id", "pair", "fee_bps", "ticks_crossed"])
        .agg(
            pl.len().alias("swap_count"),
            pl.col("volume_usd").sum().alias("volume_usd"),
            pl.col("total_fees_usd").sum().alias("total_fees_usd"),
            pl.col("is_jit").sum().alias("jit_swap_count"),
        )
        .sort(["pool_id", "ticks_crossed"])
    )

    # Add per-pool total for percentage calculation
    pool_totals = agg.group_by("pool_id").agg(
        pl.col("swap_count").sum().alias("pool_total_swaps"),
        pl.col("volume_usd").sum().alias("pool_total_volume"),
    )
    agg = agg.join(pool_totals, on="pool_id")
    agg = agg.with_columns(
        (pl.col("swap_count") / pl.col("pool_total_swaps") * 100)
        .round(4)
        .alias("pct_of_pool_swaps"),
        (pl.col("volume_usd") / pl.col("pool_total_volume") * 100)
        .round(4)
        .alias("pct_of_pool_volume"),
    )

    # Cumulative count % per pool (ascending ticks_crossed)
    agg = agg.with_columns(
        pl.col("pct_of_pool_swaps")
        .cum_sum()
        .over(["pool_id"])
        .round(4)
        .alias("cumulative_pct_swaps"),
    )

    return agg.drop(["pool_total_swaps", "pool_total_volume"])


def compute_summary_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Per-pool summary: mean, median, p95, p99, max ticks_crossed."""
    return (
        df.group_by(["pool_id", "pair", "fee_bps"])
        .agg(
            pl.col("ticks_crossed").mean().round(4).alias("mean_ticks"),
            pl.col("ticks_crossed").median().alias("median_ticks"),
            pl.col("ticks_crossed").quantile(0.95).alias("p95_ticks"),
            pl.col("ticks_crossed").quantile(0.99).alias("p99_ticks"),
            pl.col("ticks_crossed").max().alias("max_ticks"),
            pl.len().alias("total_swaps"),
            (pl.col("ticks_crossed") == 0).sum().alias("intra_tick_swaps"),
            (pl.col("ticks_crossed") > 1).sum().alias("multi_tick_swaps"),
        )
        .with_columns(
            (pl.col("intra_tick_swaps") / pl.col("total_swaps") * 100)
            .round(2)
            .alias("intra_tick_pct"),
            (pl.col("multi_tick_swaps") / pl.col("total_swaps") * 100)
            .round(2)
            .alias("multi_tick_pct"),
        )
        .sort("pool_id")
    )


@click.command()
@click.option("--pool", "pool_id", default=None, help="Process a single pool ID.")
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output CSV path (default: output/analysis/tick_crossing_distribution.csv).",
)
@click.option(
    "--no-jit",
    "no_jit",
    is_flag=True,
    default=False,
    help="Use counterfactual tick crossings for JIT swaps (as if JIT liquidity were absent).",
)
def main(pool_id: str | None, out_path: str | None, no_jit: bool) -> None:
    """Produce tick-crossing distribution from processed swap data."""
    if pool_id:
        if pool_id not in POOL_BY_ID:
            click.echo(f"Unknown pool_id '{pool_id}'. Valid: {list(POOL_BY_ID)}")
            raise SystemExit(1)
        cfgs = [POOL_BY_ID[pool_id]]
    else:
        cfgs = ALL_POOLS

    frames = [f for cfg in cfgs if (f := _load_swaps(cfg, no_jit=no_jit)) is not None]
    if not frames:
        click.echo("No processed data found. Run `jit-process --all` first.")
        raise SystemExit(1)

    combined = pl.concat(frames)
    dist = compute_distribution(combined)
    stats = compute_summary_stats(combined)

    # Output paths
    out_dir = Path(out_path).parent if out_path else OUTPUT_ROOT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_no_jit" if no_jit else ""
    dist_path = (
        Path(out_path)
        if out_path
        else out_dir / f"tick_crossing_distribution{suffix}.csv"
    )
    stats_path = dist_path.parent / (dist_path.stem + "_summary.csv")

    dist.write_csv(dist_path)
    stats.write_csv(stats_path)

    click.echo(f"Distribution → {dist_path}")
    click.echo(f"Summary      → {stats_path}")

    # Print summary to console
    click.echo("\nPer-pool summary:")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=160):
        click.echo(str(stats))


if __name__ == "__main__":
    main()
