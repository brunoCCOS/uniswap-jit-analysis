"""
full_swap_optimum.py — "what if we ran the analytical optimizer on EVERY swap?"

The production pipeline (src/enricher.py) only invokes the optimizers on swaps
the sandwich detector flagged `is_jit == True` (because that is the only case
where a real on-chain JIT capital budget — `jit_liquidity_usd` — exists to
benchmark against). This script asks a different question: across the WHOLE
swap stream (attacked or not), how often would JIT-ing have been profitable
at all, and how big is the total "available" opportunity compared to what
real on-chain JIT bots actually captured?

Method
------
We replay each pool's raw event log exactly like `src.enricher.enrich_pool`
does (same block-ordered walk over initialize/mint/burn/swap events, same
`PoolState` tick-delta bookkeeping), but for every swap — not just detected
JIT ones — we:

  1. Run the same counterfactual "no-JIT" simulation
     (`src.enricher._simulate_without_jit`) that the production pipeline uses
     to pick the search range for the optimizer.
  2. Call `src.optimizers.run_analytical_optimizer` (the CLOSED-FORM /
     Lemma 5.1-5.2 optimizer, called "the analytical optimizer" / "[28]'s
     approach" / `kh_*` columns elsewhere in this repo) to score the best
     possible JIT position for that swap.

We deliberately do NOT run the combinatorial/simulation optimizer here
(`run_simulation_optimizer`) — it is much slower and the task only needs the
fast closed-form estimate to size the opportunity.

`_build_jit_objects` (inside `src.optimizers`) needs a `JITSandwich`-shaped
object to (a) reconstruct the passive-liquidity profile with the real JIT's
contribution subtracted out, and (b) score the "actual" realized position.
For swaps with no real on-chain JIT we pass a synthetic PLACEHOLDER
`JITSandwich` with `jit_liquidity = 0`: subtracting zero liquidity is a
no-op for (a), and we simply ignore the (meaningless) "actual" score for
non-attacked swaps — only `kh_optimal_utility_usd` is used from the result.

Budget choice
-------------
`run_analytical_optimizer` needs a capital `budget` to optimize over. For
real JIT swaps the production pipeline uses the REAL on-chain JIT LP's
deployed capital (`jit_liquidity_usd`). There is no such reference for the
~97-99% of swaps that were never attacked, so this script uses a
SWAP-PROPORTIONAL PLACEHOLDER: `budget = volume_usd` (the swap's own trade
size in USD). This is applied uniformly to every swap (including the
previously-detected JIT ones) so that the resulting profit distribution is
computed on a single, consistent basis across the whole dataset — it is NOT
meant to be compared number-for-number against the budget-matched
`kh_optimal_utility_usd` column already in `swaps_enriched.parquet`.

Scope
-----
Restricted to the two SMALLER pools (2697585: 5,090 swaps; 2697647: 15,498
swaps) because this means re-running the optimizer on every swap instead of
only the handful flagged as JIT. 2697600 and 2697765 are skipped as too
large for a full per-swap re-run (see SKIPPED_POOLS below). Pool 2697588 is
excluded entirely: its optimizer run hangs on one oversized swap in the
combinatorial optimizer (known, separate, unfixed bug), and its on-disk
output/2697588/ is STALE (predates the Oct 8 analytical-optimizer
bugfixes).

Outputs
-------
  output/reports/full_swap_optimum.md
  output/reports/plots/full_swap_optimum_hist.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import POOL_BY_ID, PoolConfig  # noqa: E402
from src.detector import JITSandwich  # noqa: E402
from src.enricher import _int, _int_liq, _simulate_without_jit, _walk_segments  # noqa: E402
from src.loader import load_events  # noqa: E402
from src.optimizers import MAX_OPT_K, run_analytical_optimizer  # noqa: E402
from src.price import Q96, parse_sqrt_x96  # noqa: E402
from src.state import PoolState  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "reports"
OUT_MD = OUT_DIR / "full_swap_optimum.md"
OUT_PLOT = OUT_DIR / "plots" / "full_swap_optimum_hist.png"

INCLUDED_POOL_IDS = ["2697585", "2697647"]

SKIPPED_POOLS = {
    "2697600": "USDC/WETH fee=30bps — too large for a full per-swap optimizer re-run; skipped per task scope.",
    "2697765": "USDC/WETH fee=5bps — too large (1.18M swaps, 174MB parquet) for a full per-swap optimizer re-run; skipped per task scope.",
}

EXCLUDED_POOL_NOTE = (
    "Pool 2697588 (USDC/USDT, fee=5bps) is excluded entirely: its optimizer run "
    "hangs on one oversized swap in the combinatorial optimizer (known, separate, "
    "unfixed bug), and its on-disk output/2697588/ is STALE (predates the Oct 8 "
    "analytical-optimizer bugfixes)."
)

BUDGET_USD_DOC = (
    "budget = volume_usd (the swap's own trade size), applied uniformly to every "
    "swap including previously-detected JIT ones, since no real on-chain JIT "
    "capital exists for the vast majority of (non-attacked) swaps."
)


# ──────────────────────────────────────────────────────────────────────────────
# Replay driver: same block/event walk as src.enricher.enrich_pool, but the
# analytical optimizer is invoked for EVERY swap, not only detected-JIT ones.
# ──────────────────────────────────────────────────────────────────────────────


def _make_placeholder_jit(block_number: int, initial_tick: int, tick_spacing: int) -> JITSandwich:
    """
    Synthetic zero-liquidity JITSandwich. Exists purely so we can reuse
    run_analytical_optimizer's cross-repo object-construction path
    (src.optimizers._build_jit_objects) for swaps that never had a real
    on-chain JIT position. jit_liquidity=0 makes both the passive-liquidity
    reconstruction and the "actual position" scoring no-ops; only the
    returned kh_optimal_utility_usd is used by this script.
    """
    lower = (initial_tick // tick_spacing) * tick_spacing
    return JITSandwich(
        block_number=block_number,
        owner="",
        mint_tx="",
        mint_tx_index=0,
        burn_tx="",
        burn_tx_index=0,
        tick_lower=lower,
        tick_upper=lower + tick_spacing,
        jit_liquidity=0,
        burn_liquidity=0,
        jit_type="none",
        new_passive_liq=0,
    )


def _handle_swap_full(row: dict, state: PoolState, cfg: PoolConfig, out: list[dict]) -> None:
    initial_tick = state.tick
    initial_sqrt = state.sqrt_x96
    parsed_tick = _int(row.get("tick"))
    final_tick = parsed_tick if parsed_tick is not None else initial_tick
    final_sqrt = parse_sqrt_x96(row.get("sqrtPriceX96"))
    parsed_liq = _int(row.get("liquidity"))
    reported_liq = parsed_liq if parsed_liq is not None else state.active_liq

    direction_up = (
        (final_sqrt > initial_sqrt)
        if (initial_sqrt and final_sqrt)
        else (final_tick > initial_tick)
    )

    dec0, dec1 = cfg.token0_decimals, cfg.token1_decimals
    a0 = float(row.get("amount0") or 0)
    a1 = float(row.get("amount1") or 0)
    p0 = float(row.get("token0_price_usd") or 0)
    p1 = float(row.get("token1_price_usd") or 0)

    if a0 > 0:
        volume_usd = (a0 / 10**dec0) * p0
    elif a1 > 0:
        volume_usd = (a1 / 10**dec1) * p1
    else:
        volume_usd = 0.0

    fee_rate = cfg.fee_millionths / 1_000_000
    active_liq_start = state.active_liq
    placeholder_jit = _make_placeholder_jit(int(row["block_number"]), initial_tick, cfg.tick_spacing)

    # Tick-segment walk — mutates `state` via cross_tick_up/down exactly as
    # the production enricher does. jit=None here is already how enricher.py
    # calls this for non-JIT swaps (it's a no-op for segment accounting).
    segs = _walk_segments(
        state=state,
        initial_sqrt=initial_sqrt,
        final_sqrt=final_sqrt,
        initial_tick=initial_tick,
        final_tick=final_tick,
        direction_up=direction_up,
        jit=None,
        cfg=cfg,
        fee_rate=fee_rate,
        p0=p0,
        p1=p1,
        tx_hash=row["transaction_hash"],
        segment_records=[],
    )
    total_seg_vol = sum(s["segment_volume_usd"] for s in segs)

    if total_seg_vol > 0:
        if direction_up and p1 > 0:
            cfact_net_input = int(total_seg_vol * 10**dec1 / p1)
        elif not direction_up and p0 > 0:
            cfact_net_input = int(total_seg_vol * 10**dec0 / p0)
        else:
            cfact_net_input = (
                int(int(a1) * (1 - fee_rate)) if direction_up else int(int(a0) * (1 - fee_rate))
            )
    else:
        cfact_net_input = (
            int(int(a1) * (1 - fee_rate)) if direction_up else int(int(a0) * (1 - fee_rate))
        )

    kh_optimal_utility_usd = None
    skip_reason = None

    if initial_sqrt and cfact_net_input > 0:
        no_jit_sqrt, _ = _simulate_without_jit(
            state=state,
            initial_sqrt=initial_sqrt,
            initial_tick=initial_tick,
            initial_active_liq=active_liq_start,
            direction_up=direction_up,
            net_input=cfact_net_input,
            jit=placeholder_jit,
            tx_hash=row["transaction_hash"],
            dec0=dec0,
            dec1=dec1,
            p0=p0,
            p1=p1,
            fee_rate=fee_rate,
        )
        no_jit_tick = (
            math.floor(2 * math.log(no_jit_sqrt / Q96) / math.log(1.0001))
            if no_jit_sqrt and no_jit_sqrt > 0
            else initial_tick
        )

        tick_span_k = abs(no_jit_tick - initial_tick) // cfg.tick_spacing
        if tick_span_k > MAX_OPT_K:
            skip_reason = "tick_span_exceeds_max_opt_k"
        else:
            kh = run_analytical_optimizer(
                state=state,
                initial_sqrt=initial_sqrt,
                initial_tick=initial_tick,
                active_liq_start=active_liq_start,
                jit=placeholder_jit,
                cfg=cfg,
                p0=p0,
                p1=p1,
                direction_up=direction_up,
                amount_in_raw=a1 if direction_up else a0,
                no_jit_final_tick=no_jit_tick,
                budget=volume_usd,
            )
            if kh is None:
                skip_reason = "optimizer_returned_none"
            else:
                kh_optimal_utility_usd = kh["kh_optimal_utility_usd"]
    else:
        skip_reason = "zero_net_input"

    out.append(
        {
            "block_number": int(row["block_number"]),
            "transaction_hash": row["transaction_hash"],
            "volume_usd": volume_usd,
            "full_kh_optimal_utility_usd": kh_optimal_utility_usd,
            "skip_reason": skip_reason,
        }
    )

    state.apply_swap(final_tick, final_sqrt, reported_liq)


def run_full_optimizer(cfg: PoolConfig) -> pl.DataFrame:
    """Replay the whole event log for `cfg`, scoring every swap with the
    analytical optimizer (not just detected-JIT swaps)."""
    df = load_events(cfg)
    state = PoolState()
    out: list[dict] = []

    all_rows = df.to_dicts()
    blocks: dict[int, list[dict]] = {}
    for row in all_rows:
        bn = int(row["block_number"])
        blocks.setdefault(bn, []).append(row)

    for bn in sorted(blocks.keys()):
        for row in blocks[bn]:
            event = row["event"]
            if event == "initialize":
                state.sqrt_x96 = parse_sqrt_x96(row.get("sqrtPriceX96"))
                state.tick = _int(row.get("tick")) or 0
            elif event == "mint":
                tl = _int(row.get("tickLower"))
                tu = _int(row.get("tickUpper"))
                liq = _int_liq(row.get("liquidity"))
                if tl is not None and tu is not None and liq:
                    state.apply_mint(tl, tu, liq)
            elif event == "burn":
                tl = _int(row.get("tickLower"))
                tu = _int(row.get("tickUpper"))
                liq = _int_liq(row.get("liquidity"))
                if tl is not None and tu is not None and liq:
                    state.apply_burn(tl, tu, liq)
            elif event == "swap":
                _handle_swap_full(row, state, cfg, out)

    return pl.from_dicts(out, infer_schema_length=None) if out else pl.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Stats / report
# ──────────────────────────────────────────────────────────────────────────────


def _dist_stats(vals: np.ndarray) -> dict:
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)) if vals.size else float("nan"),
        "median": float(np.median(vals)) if vals.size else float("nan"),
        "p90": float(np.percentile(vals, 90)) if vals.size else float("nan"),
        "p99": float(np.percentile(vals, 99)) if vals.size else float("nan"),
        "max": float(np.max(vals)) if vals.size else float("nan"),
        "sum": float(np.sum(vals)) if vals.size else 0.0,
        "pct_positive": float(np.mean(vals > 0) * 100) if vals.size else float("nan"),
    }


def _actual_realized_jit_profit_usd(pool_id: str) -> tuple[float, int]:
    """
    Sum of actual_utility_usd over is_jit==True rows in
    output/<pool_id>/swaps_enriched.parquet — the already-computed
    "what the real on-chain JIT LP actually scored" baseline (this is the
    per-swap column backing the jit_sandwiches.parquet detections; the
    sandwich table itself does not carry a per-swap USD utility column).
    """
    swaps_path = REPO_ROOT / "output" / pool_id / "swaps_enriched.parquet"
    df = pl.read_parquet(swaps_path, columns=["is_jit", "actual_utility_usd"])
    jit_rows = df.filter(pl.col("is_jit") == True)  # noqa: E712
    total = jit_rows["actual_utility_usd"].drop_nulls().sum()
    return float(total or 0.0), jit_rows.height


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    per_pool_results: dict[str, dict] = {}
    all_profit_values: list[np.ndarray] = []

    for pool_id in INCLUDED_POOL_IDS:
        cfg = POOL_BY_ID[pool_id]
        print(f"[{pool_id}] replaying + scoring every swap with the analytical optimizer …")
        full_df = run_full_optimizer(cfg)

        n_swaps = full_df.height
        scored = full_df.filter(pl.col("full_kh_optimal_utility_usd").is_not_null())
        skipped = full_df.filter(pl.col("full_kh_optimal_utility_usd").is_null())
        vals = scored["full_kh_optimal_utility_usd"].to_numpy()
        stats = _dist_stats(vals)

        actual_total, jit_count = _actual_realized_jit_profit_usd(pool_id)

        per_pool_results[pool_id] = {
            "cfg": cfg,
            "n_swaps": n_swaps,
            "n_scored": scored.height,
            "n_skipped": skipped.height,
            "stats": stats,
            "actual_total_usd": actual_total,
            "jit_count": jit_count,
        }
        all_profit_values.append(vals)
        print(
            f"[{pool_id}] done: {n_swaps} swaps, {scored.height} scored, "
            f"{skipped.height} skipped, total optimal profit=${stats['sum']:,.2f}"
        )

    combined_vals = np.concatenate(all_profit_values) if all_profit_values else np.array([])
    combined_stats = _dist_stats(combined_vals)
    combined_n_swaps = sum(r["n_swaps"] for r in per_pool_results.values())
    combined_n_scored = sum(r["n_scored"] for r in per_pool_results.values())
    combined_n_skipped = sum(r["n_skipped"] for r in per_pool_results.values())
    combined_actual_total = sum(r["actual_total_usd"] for r in per_pool_results.values())
    combined_jit_count = sum(r["jit_count"] for r in per_pool_results.values())

    _write_plot(combined_vals)
    _write_markdown(per_pool_results, combined_stats, combined_n_swaps, combined_n_scored,
                     combined_n_skipped, combined_actual_total, combined_jit_count)

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_PLOT}")


def _write_plot(vals: np.ndarray) -> None:
    """
    Log-binned histogram of optimal profit. The analytical optimizer never
    returns a negative utility (it can always choose to deploy zero
    liquidity when nothing is profitable), so the distribution is bounded at
    0 from below -- the near-zero mass described in the task is a mix of an
    exact-zero spike (optimizer chooses not to provide liquidity) plus a lot
    of tiny-but-positive profits, shown here as a separate grey '= 0' bar
    next to a log-spaced histogram of the strictly-positive tail.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    zero_count = int(np.sum(vals == 0))
    pos = vals[vals > 0]

    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
    ax.hist(pos, bins=bins, color="#3b6fb5", edgecolor="none", label="optimal profit > 0")

    zero_bar_x = bins[0] / 2
    ax.bar(
        zero_bar_x,
        zero_count,
        width=zero_bar_x * 1.2,
        color="#888888",
        label="optimal profit = 0 (optimizer deploys nothing)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analytical-optimizer optimal profit per swap, USD (log scale; grey bar = exact zero)")
    ax.set_ylabel("Swap count (log scale)")
    ax.set_title(
        "Optimal JIT profit across ALL swaps — combined 2697585 + 2697647\n"
        "(budget = swap's own volume_usd; closed-form/analytical optimizer)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def _write_markdown(
    per_pool_results: dict[str, dict],
    combined_stats: dict,
    combined_n_swaps: int,
    combined_n_scored: int,
    combined_n_skipped: int,
    combined_actual_total: float,
    combined_jit_count: int,
) -> None:
    lines: list[str] = []
    lines.append("# Full-swap optimum: how often would JIT-ing have been profitable?")
    lines.append("")
    lines.append(
        "Runs the **analytical / closed-form optimizer** (`kh_*` columns elsewhere in "
        "this repo, a.k.a. \"the analytical optimizer\" / \"[28]'s approach\") on "
        "**every swap** in a pool — not only the swaps the sandwich detector flagged "
        "`is_jit == True` — to estimate how often JIT-ing would have been profitable "
        "at all, and how large the total opportunity is."
    )
    lines.append("")
    lines.append(f"**Budget convention:** {BUDGET_USD_DOC}")
    lines.append("")
    lines.append("**Scope:** restricted to the two smaller pools because this means")
    lines.append("re-running the optimizer on every swap, not just detected JIT swaps:")
    lines.append("")
    for pid, reason in SKIPPED_POOLS.items():
        lines.append(f"- SKIPPED `{pid}`: {reason}")
    lines.append(f"- EXCLUDED: {EXCLUDED_POOL_NOTE}")
    lines.append("")

    lines.append("## Per-pool results")
    lines.append("")
    lines.append(
        "| Pool | Pair | Fee | # swaps | # scored | # skipped | % profitable (of scored) | "
        "mean | median | p90 | p99 | max | **total optimal profit** | actual realized JIT profit "
        "(real bots, `is_jit`) | capture ratio (actual/optimal) |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pid, r in per_pool_results.items():
        cfg = r["cfg"]
        s = r["stats"]
        capture = (r["actual_total_usd"] / s["sum"] * 100) if s["sum"] else float("nan")
        lines.append(
            f"| {pid} | {cfg.pair_label} | {cfg.fee_millionths/100:.0f}bps | "
            f"{r['n_swaps']:,} | {r['n_scored']:,} | {r['n_skipped']:,} | "
            f"{s['pct_positive']:.2f}% | ${s['mean']:,.2f} | ${s['median']:,.2f} | "
            f"${s['p90']:,.2f} | ${s['p99']:,.2f} | ${s['max']:,.2f} | "
            f"${s['sum']:,.2f} | ${r['actual_total_usd']:,.2f} ({r['jit_count']} JIT swaps) | "
            f"{capture:.2f}% |"
        )
    combined_capture = (
        combined_actual_total / combined_stats["sum"] * 100 if combined_stats["sum"] else float("nan")
    )
    lines.append(
        f"| **combined** | — | — | {combined_n_swaps:,} | {combined_n_scored:,} | "
        f"{combined_n_skipped:,} | {combined_stats['pct_positive']:.2f}% | "
        f"${combined_stats['mean']:,.2f} | ${combined_stats['median']:,.2f} | "
        f"${combined_stats['p90']:,.2f} | ${combined_stats['p99']:,.2f} | "
        f"${combined_stats['max']:,.2f} | **${combined_stats['sum']:,.2f}** | "
        f"${combined_actual_total:,.2f} ({combined_jit_count} JIT swaps) | {combined_capture:.2f}% |"
    )
    lines.append("")
    lines.append(
        "`% profitable` = share of scored swaps where the analytical optimizer's "
        "optimal profit (given budget = that swap's own volume_usd) is strictly "
        "positive. `# skipped` = swaps where the optimizer could not be run or "
        "returned no result (e.g. the no-JIT counterfactual simulation needed to "
        "pick a search range spans more tick-ranges than `MAX_OPT_K`, or the swap "
        "has zero net input); these are excluded from both the % and the "
        "distribution stats."
    )
    lines.append("")
    lines.append(
        "Note: the optimizer's profit never goes negative in this run -- across both "
        "pools the minimum observed optimal profit is exactly $0.00, because the "
        "optimizer can always fall back to deploying zero liquidity when no range is "
        "profitable. So '% profitable' is simply 100% minus the share of swaps at "
        "exactly $0 (no swaps land strictly below zero)."
    )
    lines.append("")
    lines.append(
        "`actual realized JIT profit` is read directly from the existing "
        "`output/<pool>/swaps_enriched.parquet` column `actual_utility_usd`, summed "
        "over `is_jit == True` rows (the per-swap scoring of the REAL on-chain JIT "
        "LP's position under the same simulator; this is the same per-swap data "
        "that `output/<pool>/jit_sandwiches.parquet` is aggregated from — the "
        "sandwich table itself only carries fee-level columns, not this USD-utility "
        "figure). It is NOT recomputed by this script and uses the real, "
        "budget-matched (`jit_liquidity_usd`) optimizer runs from the main "
        "pipeline — it is therefore on a different budget basis than the "
        "volume_usd-placeholder `total optimal profit` column, so the `capture "
        "ratio` is only a rough, order-of-magnitude sense of how much of the "
        "'available' JIT opportunity (sized relative to trade size) real on-chain "
        "bots actually captured (sized relative to their real deployed capital), "
        "not an apples-to-apples ratio."
    )
    lines.append("")
    lines.append("## Plot")
    lines.append("")
    lines.append("![Optimal profit histogram](plots/full_swap_optimum_hist.png)")
    lines.append("")
    lines.append(
        "Histogram of the analytical optimizer's optimal profit across ALL scored "
        "swaps in the two included pools combined. The grey bar is the count of "
        "swaps with optimal profit of exactly $0 (optimizer deploys no liquidity); "
        "the blue bars are a log-spaced histogram of the strictly-positive tail, "
        "with both axes on a log scale, since the overwhelming majority of swaps "
        "cluster at or near zero profit with a long positive tail stretching to "
        "several thousand dollars."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
