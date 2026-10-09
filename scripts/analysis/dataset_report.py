"""
Dataset descriptive report.

Reads the processed `swaps_enriched.parquet` + `metadata.json` for the four
pools with fresh (post Oct-8-bugfix) completed output and produces a
long-form markdown report covering, per pool: pair, fee tier, event/swap
counts, JIT sandwich counts and rate, date range, total volume, and
median/p95 swap size, plus an aggregated "overall dataset" summary and a
final per-pool summary table.

Pool 2697588 (USDC/USDT, fee=5bps) is intentionally excluded: its optimizer
run hangs on one oversized swap in the combinatorial optimizer (a known,
separate, unfixed bug), and its on-disk output/2697588/ predates the Oct 8
analytical-optimizer bugfixes (stale).

Uses polars lazy scans throughout so the large 2697765 pool (1.18M swaps,
174MB parquet) is never fully materialized into pandas.

Usage:
    ./.venv/bin/python3 scripts/analysis/dataset_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import POOL_BY_ID  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "output"
REPORT_PATH = OUTPUT_ROOT / "reports" / "dataset_report.md"

INCLUDED_POOL_IDS = ["2697585", "2697600", "2697647", "2697765"]
EXCLUDED_POOL_ID = "2697588"
EXCLUDED_REASON = (
    "excluded, pending a separate fix for a combinatorial-optimizer hang on an "
    "outlier-sized swap"
)


def load_pool_stats(pool_id: str) -> dict:
    """Compute summary stats for one pool using a lazy polars scan."""
    pool_dir = OUTPUT_ROOT / pool_id
    metadata = json.loads((pool_dir / "metadata.json").read_text())
    cfg = POOL_BY_ID[pool_id]

    lf = pl.scan_parquet(pool_dir / "swaps_enriched.parquet")

    agg = lf.select(
        pl.col("timestamp").min().alias("min_ts"),
        pl.col("timestamp").max().alias("max_ts"),
        pl.col("volume_usd").sum().alias("total_volume_usd"),
        pl.col("volume_usd").median().alias("median_volume_usd"),
        pl.col("volume_usd").quantile(0.95, interpolation="linear").alias(
            "p95_volume_usd"
        ),
        pl.len().alias("n_rows"),
    ).collect()

    row = agg.row(0, named=True)

    total_swaps = metadata["total_swaps"]
    jit_count = metadata["jit_count"]
    jit_rate_pct = (jit_count / total_swaps * 100.0) if total_swaps else 0.0

    return {
        "pool_id": pool_id,
        "pair": cfg.pair_label,
        "fee_bps": metadata["fee_bps"],
        "total_events": metadata["total_events"],
        "total_swaps": total_swaps,
        "jit_count": jit_count,
        "jit_rate_pct": jit_rate_pct,
        "min_ts": row["min_ts"],
        "max_ts": row["max_ts"],
        "total_volume_usd": row["total_volume_usd"],
        "median_volume_usd": row["median_volume_usd"],
        "p95_volume_usd": row["p95_volume_usd"],
        "n_rows_in_parquet": row["n_rows"],
    }


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def qualitative_note(pool_id: str, stats: dict, all_stats: list[dict]) -> str:
    """One hand-written qualitative paragraph per pool, comparing it to the others."""
    max_volume_pool = max(all_stats, key=lambda s: s["total_volume_usd"])["pool_id"]
    min_volume_pool = min(all_stats, key=lambda s: s["total_volume_usd"])["pool_id"]
    max_jit_rate_pool = max(all_stats, key=lambda s: s["jit_rate_pct"])["pool_id"]
    min_jit_rate_pool = min(all_stats, key=lambda s: s["jit_rate_pct"])["pool_id"]
    max_swaps_pool = max(all_stats, key=lambda s: s["total_swaps"])["pool_id"]

    notes = []
    if pool_id == max_volume_pool:
        notes.append("the largest pool in this dataset by total swap volume")
    if pool_id == min_volume_pool:
        notes.append("the smallest pool in this dataset by total swap volume")
    if pool_id == max_jit_rate_pool:
        notes.append("the highest JIT-sandwich rate among the four included pools")
    if pool_id == min_jit_rate_pool:
        notes.append("the lowest JIT-sandwich rate among the four included pools")
    if pool_id == max_swaps_pool:
        notes.append(
            "by far the most heavily traded pool by swap count, roughly an order "
            "of magnitude more swaps than the next largest"
        )

    base = (
        f"Pool {pool_id} ({stats['pair']}, {stats['fee_bps']:.0f}bps fee tier) "
        f"recorded {stats['total_swaps']:,} decoded swaps out of "
        f"{stats['total_events']:,} total on-chain events ingested, spanning "
        f"{stats['min_ts']} to {stats['max_ts']}. Of those swaps, "
        f"{stats['jit_count']:,} ({stats['jit_rate_pct']:.2f}%) were flagged as "
        f"JIT sandwich attacks, moving a total of {fmt_usd(stats['total_volume_usd'])} "
        f"in notional swap volume (median trade size "
        f"{fmt_usd(stats['median_volume_usd'])}, p95 trade size "
        f"{fmt_usd(stats['p95_volume_usd'])})."
    )
    if notes:
        base += " This pool stands out as " + "; it is also ".join(notes) + "."
    return base


def build_report(all_stats: list[dict]) -> str:
    total_events = sum(s["total_events"] for s in all_stats)
    total_swaps = sum(s["total_swaps"] for s in all_stats)
    total_jit = sum(s["jit_count"] for s in all_stats)
    overall_jit_rate = (total_jit / total_swaps * 100.0) if total_swaps else 0.0
    total_volume = sum(s["total_volume_usd"] for s in all_stats)

    lines: list[str] = []
    lines.append("# Dataset Report")
    lines.append("")
    lines.append(
        "Descriptive overview of the four Uniswap v3 pools with fresh "
        "(post-fix, Oct 8) completed JIT-analysis output. Each pool was "
        "processed end-to-end: raw on-chain events were decoded into swaps, "
        "JIT sandwich attacks were detected, and both the combinatorial "
        "optimizer (Algorithm 1 / ours) and the analytical closed-form "
        "optimizer ([28]'s approach) were run against the same "
        "same-capital budget (`jit_liquidity_usd`) as the real on-chain JIT "
        "LP."
    )
    lines.append("")

    for pool_id in INCLUDED_POOL_IDS:
        stats = next(s for s in all_stats if s["pool_id"] == pool_id)
        lines.append(f"## Pool {pool_id} — {stats['pair']} ({stats['fee_bps']:.0f}bps)")
        lines.append("")
        lines.append(f"- **Pair:** {stats['pair']}")
        lines.append(f"- **Fee tier:** {stats['fee_bps']:.0f}bps")
        lines.append(f"- **Total on-chain events ingested:** {stats['total_events']:,}")
        lines.append(f"- **Total decoded swaps:** {stats['total_swaps']:,}")
        lines.append(
            f"- **Detected JIT sandwiches:** {stats['jit_count']:,} "
            f"({stats['jit_rate_pct']:.2f}% of swaps)"
        )
        lines.append(f"- **Date/time range:** {stats['min_ts']} → {stats['max_ts']}")
        lines.append(f"- **Total swap volume:** {fmt_usd(stats['total_volume_usd'])}")
        lines.append(
            f"- **Median swap size:** {fmt_usd(stats['median_volume_usd'])}  "
            f"**p95 swap size:** {fmt_usd(stats['p95_volume_usd'])}"
        )
        lines.append("")
        lines.append(qualitative_note(pool_id, stats, all_stats))
        lines.append("")

    lines.append("## Overall Dataset Summary")
    lines.append("")
    lines.append(
        f"Across the four included pools, the dataset covers "
        f"**{total_events:,} total on-chain events**, **{total_swaps:,} total "
        f"decoded swaps**, and **{total_jit:,} total detected JIT sandwiches** "
        f"(overall JIT rate {overall_jit_rate:.2f}%), for a combined "
        f"**{fmt_usd(total_volume)}** of swap volume."
    )
    lines.append("")
    lines.append(
        f"**Pool {EXCLUDED_POOL_ID} (USDC/USDT, fee=5bps) is {EXCLUDED_REASON}.** "
        "Its on-disk output predates the Oct 8 analytical-optimizer bugfixes and "
        "is therefore not used for any numeric results in this or companion "
        "reports."
    )
    lines.append("")

    lines.append("## Per-Pool Summary Table")
    lines.append("")
    lines.append(
        "| pool_id | pair | fee_bps | events | swaps | jit_count | "
        "jit_rate_pct | total_volume_usd | median_swap_usd |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for pool_id in INCLUDED_POOL_IDS:
        s = next(st for st in all_stats if st["pool_id"] == pool_id)
        lines.append(
            f"| {s['pool_id']} | {s['pair']} | {s['fee_bps']:.0f} | "
            f"{s['total_events']:,} | {s['total_swaps']:,} | {s['jit_count']:,} | "
            f"{s['jit_rate_pct']:.2f} | {fmt_usd(s['total_volume_usd'])} | "
            f"{fmt_usd(s['median_volume_usd'])} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    all_stats = [load_pool_stats(pool_id) for pool_id in INCLUDED_POOL_IDS]
    report = build_report(all_stats)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH} ({len(report):,} bytes)")

    for s in all_stats:
        print(
            f"  pool {s['pool_id']}: events={s['total_events']:,} "
            f"swaps={s['total_swaps']:,} jit={s['jit_count']:,} "
            f"({s['jit_rate_pct']:.2f}%) volume={fmt_usd(s['total_volume_usd'])} "
            f"median_swap={fmt_usd(s['median_volume_usd'])}"
        )


if __name__ == "__main__":
    main()
