"""
CLI entry point.

Usage:
    python -m scripts.process 2697600
    python -m scripts.process --all
    python -m scripts.process --all --parallel
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ALL_POOLS, POOL_BY_ID, PoolConfig
from src.enricher import enrich_pool
from src.loader import load_events

OUTPUT_ROOT = Path(__file__).parent.parent / "output"


def run_pool(cfg: PoolConfig) -> dict:
    t0 = time.time()
    out_dir = OUTPUT_ROOT / cfg.pool_id
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[{cfg.pool_id}] Loading {cfg.total_csv.name} …")
    df = load_events(cfg)
    click.echo(f"[{cfg.pool_id}] {len(df):,} events loaded")

    swaps_df, segments_df, jit_df = enrich_pool(df, cfg)

    swaps_path = out_dir / "swaps_enriched.parquet"
    segs_path = out_dir / "swap_tick_segments.parquet"
    jit_path = out_dir / "jit_sandwiches.parquet"

    if len(swaps_df) > 0:
        swaps_df.write_parquet(swaps_path)
    if len(segments_df) > 0:
        segments_df.write_parquet(segs_path)
    if len(jit_df) > 0:
        jit_df.write_parquet(jit_path)

    elapsed = time.time() - t0
    stats = {
        "pool_id": cfg.pool_id,
        "pair": cfg.pair_label,
        "fee_bps": cfg.fee_millionths / 100,
        "total_events": len(df),
        "total_swaps": len(swaps_df),
        "jit_count": len(jit_df),
        "elapsed_s": round(elapsed, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2))
    click.echo(
        f"[{cfg.pool_id}] Done — {stats['total_swaps']:,} swaps, "
        f"{stats['jit_count']} JIT sandwiches ({elapsed:.1f}s)"
    )
    return stats


@click.command()
@click.argument("pool_id", required=False)
@click.option("--all", "process_all", is_flag=True, help="Process all pools sequentially.")
@click.option("--parallel", is_flag=True, help="Process all pools in parallel (with --all).")
def main(pool_id: str | None, process_all: bool, parallel: bool) -> None:
    """Enrich Uniswap V3 pool event data with JIT detection and fee attribution."""
    if process_all:
        pools = ALL_POOLS
    elif pool_id:
        if pool_id not in POOL_BY_ID:
            click.echo(f"Unknown pool_id '{pool_id}'. Valid: {list(POOL_BY_ID)}")
            raise SystemExit(1)
        pools = [POOL_BY_ID[pool_id]]
    else:
        click.echo("Provide a pool_id or --all. Use --help for options.")
        raise SystemExit(1)

    if parallel and len(pools) > 1:
        with ProcessPoolExecutor(max_workers=len(pools)) as ex:
            futures = {ex.submit(run_pool, cfg): cfg.pool_id for cfg in pools}
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    click.echo(f"[{pid}] FAILED: {exc}", err=True)
    else:
        for cfg in pools:
            run_pool(cfg)


if __name__ == "__main__":
    main()
