"""
Main enrichment pipeline: sequential event walk producing per-swap and per-tick-segment records.
"""

from __future__ import annotations

import polars as pl
from tqdm import tqdm

from src.config import PoolConfig
from src.detector import JITSandwich, detect_jit
from src.price import parse_sqrt_x96, segment_amounts, sqrt_x96_to_price, tick_to_sqrt_x96
from src.state import PoolState

# ──────────────────────────────────────────────────────────────────────────────
# Output record types (plain dicts for fast accumulation, then one bulk DataFrame)
# ──────────────────────────────────────────────────────────────────────────────


def _empty_swap_record() -> dict:
    return {
        "block_number": None,
        "timestamp": None,
        "transaction_hash": None,
        "transaction_index": None,
        "log_index": None,
        "sender_address": None,
        "recipient_address": None,
        "txFrom": None,
        "amount0": None,
        "amount1": None,
        "token0_price_usd": None,
        "token1_price_usd": None,
        "volume_usd": None,
        "initial_sqrt_x96": None,
        "final_sqrt_x96": None,
        "initial_price": None,
        "final_price": None,
        "price_impact_pct": None,
        "initial_tick": None,
        "final_tick": None,
        "ticks_crossed": None,
        "direction": None,
        "active_liq_start": None,
        "active_liq_end": None,
        "jit_liquidity_weighted": None,
        "passive_liquidity_weighted": None,
        "jit_fraction_weighted": None,
        "is_jit": None,
        "jit_type": None,
        "total_fees_usd": None,
        "fees_to_jit_usd": None,
        "fees_to_passive_usd": None,
        "jit_mint_tx": None,
        "jit_burn_tx": None,
        "jit_owner": None,
        "jit_tick_lower": None,
        "jit_tick_upper": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core per-block processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_block(
    rows: list[dict],
    state: PoolState,
    cfg: PoolConfig,
    swap_records: list[dict],
    segment_records: list[dict],
    jit_records: list[dict],
) -> None:
    # Pass 1: detect JIT sandwiches in this block
    jit_map: dict[str, JITSandwich] = detect_jit(rows)

    # Pass 2: update state and emit records
    for row in rows:
        event = row["event"]

        if event == "initialize":
            sqrt_x96 = parse_sqrt_x96(row.get("sqrtPriceX96"))
            tick = _int(row.get("tick")) or 0
            state.sqrt_x96 = sqrt_x96
            state.tick = tick

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
            _handle_swap(row, state, cfg, jit_map, swap_records, segment_records)

    # Emit JIT sandwich summary records (once per block after processing)
    emitted_mints: set[str] = set()
    for sandwich in jit_map.values():
        if sandwich.mint_tx in emitted_mints:
            continue
        emitted_mints.add(sandwich.mint_tx)
        # Aggregate fees from swap records that belong to this sandwich
        total_fees = sum(
            r["total_fees_usd"] or 0
            for r in swap_records
            if r["transaction_hash"] in sandwich.swap_tx_hashes
        )
        fees_to_jit = sum(
            r["fees_to_jit_usd"] or 0
            for r in swap_records
            if r["transaction_hash"] in sandwich.swap_tx_hashes
        )
        vol = sum(
            r["volume_usd"] or 0
            for r in swap_records
            if r["transaction_hash"] in sandwich.swap_tx_hashes
        )
        jit_records.append(
            {
                "sandwich_id": sandwich.mint_tx,
                "block_number": sandwich.block_number,
                "timestamp": rows[0]["timestamp"],
                "owner": sandwich.owner,
                "mint_tx": sandwich.mint_tx,
                "burn_tx": sandwich.burn_tx,
                "tick_lower": sandwich.tick_lower,
                "tick_upper": sandwich.tick_upper,
                "jit_liquidity": sandwich.jit_liquidity,
                "burn_liquidity": sandwich.burn_liquidity,
                "jit_type": sandwich.jit_type,
                "new_passive_liq": sandwich.new_passive_liq,
                "swap_count": len(sandwich.swap_tx_hashes),
                "total_volume_usd": vol,
                "fees_captured_usd": fees_to_jit,
                "fees_missed_by_passive_usd": fees_to_jit,
                "total_fees_usd": total_fees,
            }
        )


def _handle_swap(
    row: dict,
    state: PoolState,
    cfg: PoolConfig,
    jit_map: dict[str, JITSandwich],
    swap_records: list[dict],
    segment_records: list[dict],
) -> None:
    initial_tick = state.tick
    initial_sqrt = state.sqrt_x96
    final_tick = _int(row.get("tick")) or initial_tick
    final_sqrt = parse_sqrt_x96(row.get("sqrtPriceX96"))
    reported_liq = _int_liq(row.get("liquidity")) or state.active_liq

    direction_up = final_tick > initial_tick
    direction = "buy" if direction_up else "sell"

    # Prices
    dec0, dec1 = cfg.token0_decimals, cfg.token1_decimals
    initial_price = sqrt_x96_to_price(initial_sqrt, dec0, dec1) if initial_sqrt else None
    final_price = sqrt_x96_to_price(final_sqrt, dec0, dec1) if final_sqrt else None

    price_impact = None
    if initial_price and final_price and initial_price != 0:
        price_impact = (final_price - initial_price) / initial_price * 100.0

    # Volume in USD (use input token amount)
    a0 = float(row.get("amount0") or 0)
    a1 = float(row.get("amount1") or 0)
    p0 = float(row.get("token0_price_usd") or 0)
    p1 = float(row.get("token1_price_usd") or 0)

    if a0 > 0:
        volume_usd = (a0 / 10**dec0) * p0
    else:
        volume_usd = (abs(a1) / 10**dec1) * p1

    # JIT lookup
    jit = jit_map.get(row["transaction_hash"])
    jit_liq_at_start = jit.jit_liquidity if jit else 0

    # ── Per-tick segment walk ──────────────────────────────────────────────
    segs = _walk_segments(
        state=state,
        initial_sqrt=initial_sqrt,
        final_sqrt=final_sqrt,
        initial_tick=initial_tick,
        final_tick=final_tick,
        direction_up=direction_up,
        jit=jit,
        cfg=cfg,
        p0=p0,
        p1=p1,
        tx_hash=row["transaction_hash"],
        segment_records=segment_records,
    )

    # ── Aggregate across segments ─────────────────────────────────────────
    total_seg_vol = sum(s["segment_volume_usd"] for s in segs)

    # Scale fees to match actual swap volume (segment math can diverge slightly
    # from on-chain amounts due to floating-point price approximations)
    scale = volume_usd / total_seg_vol if total_seg_vol > 0 else 1.0
    fee_rate = cfg.fee_millionths / 1_000_000
    total_fees = volume_usd * fee_rate

    if segs:
        fees_to_jit = sum(s["fees_to_jit"] for s in segs) * scale
        fees_to_passive = total_fees - fees_to_jit
        jit_liq_weighted = sum(s["jit_liquidity"] * s["segment_volume_usd"] for s in segs)
        passive_liq_weighted = sum(s["passive_liquidity"] * s["segment_volume_usd"] for s in segs)
        total_vol_weighted = sum(s["segment_volume_usd"] for s in segs)
        if total_vol_weighted > 0:
            jit_liq_weighted /= total_vol_weighted
            passive_liq_weighted /= total_vol_weighted
        total_active_weighted = jit_liq_weighted + passive_liq_weighted
        jit_fraction = jit_liq_weighted / total_active_weighted if total_active_weighted > 0 else 0.0
    else:
        # No tick data available — fall back to start-state liquidity
        total_liq = reported_liq
        fees_to_jit = total_fees * (jit_liq_at_start / total_liq) if total_liq else 0
        fees_to_passive = total_fees - fees_to_jit
        jit_liq_weighted = float(jit_liq_at_start)
        passive_liq_weighted = float(total_liq - jit_liq_at_start)
        jit_fraction = jit_liq_at_start / total_liq if total_liq else 0.0

    rec = _empty_swap_record()
    rec.update(
        {
            "block_number": int(row["block_number"]),
            "timestamp": row.get("timestamp"),
            "transaction_hash": row["transaction_hash"],
            "transaction_index": int(row["transaction_index"]),
            "log_index": int(row["log_index"]),
            "sender_address": row.get("sender_address"),
            "recipient_address": row.get("recipient_address"),
            "txFrom": row.get("txFrom"),
            "amount0": a0,
            "amount1": a1,
            "token0_price_usd": p0,
            "token1_price_usd": p1,
            "volume_usd": volume_usd,
            "initial_sqrt_x96": str(initial_sqrt) if initial_sqrt else None,
            "final_sqrt_x96": str(final_sqrt) if final_sqrt else None,
            "initial_price": initial_price,
            "final_price": final_price,
            "price_impact_pct": price_impact,
            "initial_tick": initial_tick,
            "final_tick": final_tick,
            "ticks_crossed": len(segs),
            "direction": direction,
            "active_liq_start": state.active_liq,
            "active_liq_end": reported_liq,
            "jit_liquidity_weighted": jit_liq_weighted,
            "passive_liquidity_weighted": passive_liq_weighted,
            "jit_fraction_weighted": jit_fraction,
            "is_jit": jit is not None,
            "jit_type": jit.jit_type if jit else None,
            "total_fees_usd": total_fees,
            "fees_to_jit_usd": fees_to_jit,
            "fees_to_passive_usd": fees_to_passive,
            "jit_mint_tx": jit.mint_tx if jit else None,
            "jit_burn_tx": jit.burn_tx if jit else None,
            "jit_owner": jit.owner if jit else None,
            "jit_tick_lower": jit.tick_lower if jit else None,
            "jit_tick_upper": jit.tick_upper if jit else None,
        }
    )
    swap_records.append(rec)

    # Advance pool state (ground-truth liq from the on-chain report)
    state.apply_swap(final_tick, final_sqrt, reported_liq)


def _walk_segments(
    state: PoolState,
    initial_sqrt: int,
    final_sqrt: int,
    initial_tick: int,
    final_tick: int,
    direction_up: bool,
    jit: JITSandwich | None,
    cfg: PoolConfig,
    p0: float,
    p1: float,
    tx_hash: str,
    segment_records: list[dict],
) -> list[dict]:
    """
    Walk tick segments for a swap and emit one segment record per crossed tick range.
    Returns the list of segment dicts so the caller can aggregate fees.
    """
    if initial_tick == final_tick or initial_sqrt == 0 or final_sqrt == 0:
        return []

    boundaries = state.ticks_between(initial_tick, final_tick)
    if not boundaries:
        return []

    dec0, dec1 = cfg.token0_decimals, cfg.token1_decimals
    fee_rate = cfg.fee_millionths / 1_000_000

    current_sqrt = initial_sqrt
    current_tick = initial_tick
    active_liq = state.active_liq
    seg_index = 0
    result: list[dict] = []

    for i, boundary_tick in enumerate(boundaries):
        # Use actual final sqrtPrice for the last boundary; tick formula for intermediate ones.
        # This ensures the first and last segments use the real prices from the pool,
        # not an approximation from tick_to_sqrt_x96.
        is_last = (i == len(boundaries) - 1)
        boundary_sqrt = final_sqrt if is_last else tick_to_sqrt_x96(boundary_tick)
        if boundary_sqrt == 0:
            continue

        # Amount produced by the V3 formula for this segment.
        # segment_amounts returns (amount0, amount1) where the INPUT token is positive.
        amt0, amt1 = segment_amounts(current_sqrt, boundary_sqrt, active_liq, direction_up)

        # Volume = input token only (fee is charged on input; don't double-count both legs)
        if direction_up:
            seg_vol = (amt1 / 10**dec1) * p1  # token1 is input when going up
        else:
            seg_vol = (amt0 / 10**dec0) * p0  # token0 is input when going down

        # JIT liquidity active in this segment
        jit_liq = 0
        if jit and jit.tick_lower <= current_tick < jit.tick_upper:
            jit_liq = jit.jit_liquidity
        passive_liq = max(0, active_liq - jit_liq)

        seg_fees = seg_vol * fee_rate
        jit_fees = seg_fees * (jit_liq / active_liq) if active_liq > 0 else 0.0
        passive_fees = seg_fees - jit_fees

        seg = {
            "transaction_hash": tx_hash,
            "segment_index": seg_index,
            "tick_start": current_tick,
            "tick_end": boundary_tick,
            "sqrt_price_start": str(current_sqrt),
            "sqrt_price_end": str(boundary_sqrt),
            "total_liquidity": active_liq,
            "jit_liquidity": jit_liq,
            "passive_liquidity": passive_liq,
            "segment_volume_usd": seg_vol,
            "fees_total": seg_fees,
            "fees_to_jit": jit_fees,
            "fees_to_passive": passive_fees,
        }
        result.append(seg)
        segment_records.append(seg)

        # Cross the tick boundary — update liquidity for next segment
        if direction_up:
            state.cross_tick_up(boundary_tick)
        else:
            state.cross_tick_down(boundary_tick)
        active_liq = state.active_liq

        current_sqrt = boundary_sqrt
        current_tick = boundary_tick
        seg_index += 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def enrich_pool(df: pl.DataFrame, cfg: PoolConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Process all events for one pool.

    Returns:
        swaps_df       — one row per swap
        segments_df    — one row per tick segment per swap
        jit_df         — one row per JIT sandwich
    """
    state = PoolState()
    swap_records: list[dict] = []
    segment_records: list[dict] = []
    jit_records: list[dict] = []

    # Group by block, maintain order
    all_rows = df.to_dicts()

    blocks: dict[int, list[dict]] = {}
    for row in all_rows:
        bn = int(row["block_number"])
        blocks.setdefault(bn, []).append(row)

    for bn in tqdm(sorted(blocks.keys()), desc=cfg.pool_id, unit="block"):
        _process_block(
            rows=blocks[bn],
            state=state,
            cfg=cfg,
            swap_records=swap_records,
            segment_records=segment_records,
            jit_records=jit_records,
        )

    return (
        pl.from_dicts(swap_records, infer_schema_length=None) if swap_records else pl.DataFrame(),
        pl.from_dicts(segment_records, infer_schema_length=None) if segment_records else pl.DataFrame(),
        pl.from_dicts(jit_records, infer_schema_length=None) if jit_records else pl.DataFrame(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _int_liq(v) -> int:
    """Parse liquidity value (float in CSV due to nulls) to int."""
    if v is None:
        return 0
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0
