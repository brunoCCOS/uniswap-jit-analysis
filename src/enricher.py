"""
Main enrichment pipeline: sequential event walk producing per-swap and per-tick-segment records.
"""

from __future__ import annotations

import math

import polars as pl
from tqdm import tqdm

from src.config import PoolConfig
from src.detector import JITSandwich, detect_jit
from src.optimizers import run_simulation_optimizer, run_analytical_optimizer, _comb_ts_mult, MAX_OPT_K
from src.price import (
    Q96,
    parse_sqrt_x96,
    position_value_usd,
    segment_amounts,
    sqrt_x96_to_price,
    tick_to_sqrt_x96,
)
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
        "jit_liquidity_usd": None,
        "initial_tick_price": None,
        "final_tick_price": None,
        "no_jit_final_sqrt_x96": None,
        "no_jit_final_tick": None,
        "no_jit_final_price": None,
        "no_jit_price_impact_pct": None,
        "optimal_tick_lower": None,
        "optimal_tick_upper": None,
        "optimal_jit_liquidity": None,
        "optimal_utility_usd": None,
        "actual_utility_usd": None,

        "kh_range-0_tick_lower": None,
        "kh_range-0_tick_upper": None,
        "kh_range-0_jit_liquidity": None,
        "kh_range-0_optimal_utility_usd": None,
        "kh_range-0_actual_utility_usd": None,

        "kh_range-1_tick_lower": None,
        "kh_range-1_tick_upper": None,
        "kh_range-1_jit_liquidity": None,
        "kh_optimal_utility_usd": None,
        "kh_actual_utility_usd": None,
        
        "comb_ts_mult": None,
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
        swap_tx_set = set(sandwich.swap_tx_hashes)
        total_fees = fees_to_jit = vol = 0.0
        for r in swap_records:
            if r["transaction_hash"] in swap_tx_set:
                total_fees += r["total_fees_usd"] or 0
                fees_to_jit += r["fees_to_jit_usd"] or 0
                vol += r["volume_usd"] or 0
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
    _parsed_tick = _int(row.get("tick"))
    final_tick = _parsed_tick if _parsed_tick is not None else initial_tick
    final_sqrt = parse_sqrt_x96(row.get("sqrtPriceX96"))
    _parsed_liq = _int(row.get("liquidity"))
    reported_liq = _parsed_liq if _parsed_liq is not None else state.active_liq

    # Use sqrt comparison: ticks don't change for intra-tick swaps
    direction_up = (
        (final_sqrt > initial_sqrt)
        if (initial_sqrt and final_sqrt)
        else (final_tick > initial_tick)
    )
    direction = "buy" if direction_up else "sell"

    # Prices
    dec0, dec1 = cfg.token0_decimals, cfg.token1_decimals
    initial_price = (
        sqrt_x96_to_price(initial_sqrt, dec0, dec1) if initial_sqrt else None
    )
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
    elif a1 > 0:
        volume_usd = (a1 / 10**dec1) * p1
    else:
        volume_usd = 0.0

    # JIT lookup
    jit = jit_map.get(row["transaction_hash"])
    jit_liq_at_start = jit.jit_liquidity if jit else 0

    fee_rate = cfg.fee_millionths / 1_000_000

    # Capture before the walk mutates state.active_liq via tick crossings
    active_liq_start = state.active_liq

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
        fee_rate=fee_rate,
        p0=p0,
        p1=p1,
        tx_hash=row["transaction_hash"],
        segment_records=segment_records,
    )

    # ── Aggregate across segments ─────────────────────────────────────────
    total_seg_vol = sum(s["segment_volume_usd"] for s in segs)
    total_fees = volume_usd * fee_rate

    # scale = volume_usd / total_seg_vol captures two effects:
    #   1. Fee deduction: V3 moves the price with the net (post-fee) input, so
    #      our formula gives the net amount while the CSV reports gross. scale ≈ 1/(1-fee_rate).
    #   2. State machine drift: if active_liq in our tick map is lower than the
    #      on-chain value (e.g., due to large positions not in the CSV window),
    #      segment volumes are underestimated and scale >> 1.
    #
    # For JIT attribution, the JIT position is always correctly tracked (it was minted
    # in the same block). Only passive/unknown liquidity is underestimated. So:
    #   true_total_liq ≈ our_total_liq * scale
    #   jit_fraction   = jit_liq / (our_total_liq * scale)  ← scale-corrected denominator
    #
    # This correctly handles both the normal case (scale ≈ 1.003) and the drift
    # case (scale >> 1) without any special-casing.
    scale = volume_usd / total_seg_vol if total_seg_vol > 0 else 1.0

    if segs:
        jit_liq_weighted = sum(
            s["jit_liquidity"] * s["segment_volume_usd"] for s in segs
        )
        total_liq_weighted = sum(
            s["total_liquidity"] * s["segment_volume_usd"] for s in segs
        )
        if total_seg_vol > 0:
            jit_liq_weighted /= total_seg_vol
            total_liq_weighted /= total_seg_vol
        # Corrected denominator: scale-adjusted total liquidity
        corrected_total_liq = total_liq_weighted * scale
        jit_fraction = (
            jit_liq_weighted / corrected_total_liq if corrected_total_liq > 0 else 0.0
        )
        fees_to_jit = total_fees * jit_fraction
        fees_to_passive = total_fees - fees_to_jit
        # Passive liq estimate: scale corrects for untracked positions
        passive_liq_weighted = corrected_total_liq - jit_liq_weighted

        # Propagate scale to segment records so segment fees sum to swap fees.
        # Segment dicts are shared-reference with segment_records, so this updates both.
        if scale != 1.0:
            for seg in segs:
                seg["segment_volume_usd"] *= scale
                seg["fees_total"] *= scale
                seg["fees_to_jit"] *= scale
                seg["fees_to_passive"] *= scale
    else:
        # No tick data available — fall back to start-state liquidity
        total_liq = reported_liq
        fees_to_jit = total_fees * (jit_liq_at_start / total_liq) if total_liq else 0
        fees_to_passive = total_fees - fees_to_jit
        jit_liq_weighted = float(jit_liq_at_start)
        passive_liq_weighted = float(total_liq - jit_liq_at_start)
        jit_fraction = jit_liq_at_start / total_liq if total_liq else 0.0

    # ── Tick boundary prices ───────────────────────────────────────────────
    initial_tick_price = sqrt_x96_to_price(tick_to_sqrt_x96(initial_tick), dec0, dec1)
    final_tick_price = sqrt_x96_to_price(tick_to_sqrt_x96(final_tick), dec0, dec1)

    # ── JIT position USD value ─────────────────────────────────────────────
    jit_liq_usd = (
        position_value_usd(
            jit.tick_lower,
            jit.tick_upper,
            jit.jit_liquidity,
            initial_sqrt,
            initial_tick,
            dec0,
            dec1,
            p0,
            p1,
        )
        if jit
        else None
    )

    # ── Counterfactual: swap without JIT liquidity ─────────────────────────
    if jit:
        # Derive net input from total_seg_vol so it is consistent with the
        # same estimated liquidity used in the segment walk.  Using
        # a1*(1-fee_rate) can underestimate the true net when our state
        # machine overestimates pool liquidity (scale < 1/(1-fee_rate)),
        # causing the counterfactual to undershoot and show less impact than
        # the actual swap — the opposite of the correct direction.
        if total_seg_vol > 0:
            if direction_up and p1 > 0:
                cfact_net_input = int(total_seg_vol * 10**dec1 / p1)
            elif not direction_up and p0 > 0:
                cfact_net_input = int(total_seg_vol * 10**dec0 / p0)
            else:
                cfact_net_input = (
                    int(int(a1) * (1 - fee_rate))
                    if direction_up
                    else int(int(a0) * (1 - fee_rate))
                )
        else:
            cfact_net_input = (
                int(int(a1) * (1 - fee_rate))
                if direction_up
                else int(int(a0) * (1 - fee_rate))
            )

        no_jit_sqrt, no_jit_segs = _simulate_without_jit(
            state=state,
            initial_sqrt=initial_sqrt,
            initial_tick=initial_tick,
            initial_active_liq=active_liq_start,
            direction_up=direction_up,
            net_input=cfact_net_input,
            jit=jit,
            tx_hash=row["transaction_hash"],
            dec0=dec0,
            dec1=dec1,
            p0=p0,
            p1=p1,
            fee_rate=fee_rate,
        )
        segment_records.extend(no_jit_segs)
        if no_jit_sqrt and no_jit_sqrt > 0:
            no_jit_price = sqrt_x96_to_price(no_jit_sqrt, dec0, dec1)
            no_jit_tick = math.floor(2 * math.log(no_jit_sqrt / Q96) / math.log(1.0001))
            no_jit_impact = (
                (no_jit_price - initial_price) / initial_price * 100.0
                if initial_price and no_jit_price
                else None
            )
        else:
            no_jit_price = no_jit_tick = no_jit_impact = None
        no_jit_sqrt_str = str(no_jit_sqrt) if no_jit_sqrt else None

        tick_span_k = abs((no_jit_tick or initial_tick) - initial_tick) // cfg.tick_spacing
        comb_ts_mult = _comb_ts_mult(tick_span_k)
        comb_tick_spacing = cfg.tick_spacing * comb_ts_mult

        opt_args = dict(
            state=state,
            initial_sqrt=initial_sqrt,
            initial_tick=initial_tick,
            active_liq_start=active_liq_start,
            jit=jit,
            cfg=cfg,
            p0=p0,
            p1=p1,
            direction_up=direction_up,
            amount_in_raw=a1 if direction_up else a0,
            no_jit_final_tick=no_jit_tick or initial_tick,
            budget=jit_liq_usd or 0.0,
        )
        if tick_span_k > MAX_OPT_K:
            opt = None
            kh = None
            if jit is not None:
                from pprint import pprint
                pprint(f"{type(kh)}, {type(jit)}")
        else:
            opt = run_simulation_optimizer(**opt_args, comb_tick_spacing=comb_tick_spacing)
            kh = run_analytical_optimizer(**opt_args)

    else:
        no_jit_sqrt_str = no_jit_price = no_jit_tick = no_jit_impact = None
        opt = None
        kh = None
        
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
            "active_liq_start": active_liq_start,
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
            "jit_liquidity_usd": jit_liq_usd,
            "initial_tick_price": initial_tick_price,
            "final_tick_price": final_tick_price,
            "no_jit_final_sqrt_x96": no_jit_sqrt_str,
            "no_jit_final_tick": no_jit_tick,
            "no_jit_final_price": no_jit_price,
            "no_jit_price_impact_pct": no_jit_impact,
            "optimal_tick_lower": opt["optimal_tick_lower"] if opt else None,
            "optimal_tick_upper": opt["optimal_tick_upper"] if opt else None,
            "optimal_jit_liquidity": opt["optimal_jit_liquidity"] if opt else None,
            "optimal_utility_usd": opt["optimal_utility_usd"] if opt else None,
            "actual_utility_usd": opt["actual_utility_usd"] if opt else None,
            
            "kh_range-0_tick_lower": kh["kh_range-0_tick_lower"] if kh else None,
            "kh_range-0_tick_upper": kh["kh_range-0_tick_upper"] if kh else None,
            "kh_range-0_jit_liquidity": kh["kh_range-0_jit_liquidity"] if kh else None,

            "kh_range-1_tick_lower": kh["kh_range-1_tick_lower"] if kh else None,
            "kh_range-1_tick_upper": kh["kh_range-1_tick_upper"] if kh else None,
            "kh_range-1_jit_liquidity": kh["kh_range-1_jit_liquidity"] if kh else None,
            
            "kh_optimal_utility_usd": kh["kh_optimal_utility_usd"] if kh else None,
            "kh_actual_utility_usd": kh["kh_actual_utility_usd"] if kh else None,

            "comb_ts_mult": comb_ts_mult if jit else None,
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
    fee_rate: float,
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

    current_sqrt = initial_sqrt
    current_tick = initial_tick
    active_liq = state.active_liq
    seg_index = 0
    result: list[dict] = []

    for i, boundary_tick in enumerate(boundaries):
        # Use actual final sqrtPrice for the last boundary; tick formula for intermediate ones.
        # This ensures the first and last segments use the real prices from the pool,
        # not an approximation from tick_to_sqrt_x96.
        is_last = i == len(boundaries) - 1
        boundary_sqrt = final_sqrt if is_last else tick_to_sqrt_x96(boundary_tick)
        if boundary_sqrt == 0:
            continue

        # Amount produced by the V3 formula for this segment.
        # segment_amounts returns (amount0, amount1) where the INPUT token is positive.
        amt0, amt1 = segment_amounts(
            current_sqrt, boundary_sqrt, active_liq, direction_up
        )

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
            "price_start_usd": sqrt_x96_to_price(current_sqrt, dec0, dec1),
            "price_end_usd": sqrt_x96_to_price(boundary_sqrt, dec0, dec1),
            "total_liquidity": active_liq,
            "jit_liquidity": jit_liq,
            "passive_liquidity": passive_liq,
            "segment_volume_usd": seg_vol,
            "fees_total": seg_fees,
            "fees_to_jit": jit_fees,
            "fees_to_passive": passive_fees,
            "is_counterfactual": False,
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
# Counterfactual simulation (no JIT liquidity)
# ──────────────────────────────────────────────────────────────────────────────


def _jit_adjusted_delta(boundary_tick: int, state: PoolState, jit: JITSandwich) -> int:
    """
    Return the effective tick delta for a passive-only pool (JIT contribution removed).
    JIT mint added +jit_liq at tick_lower and -jit_liq at tick_upper.
    """
    base = state.tick_deltas.get(boundary_tick, 0)
    if boundary_tick == jit.tick_lower:
        return base - jit.jit_liquidity
    if boundary_tick == jit.tick_upper:
        return base + jit.jit_liquidity
    return base


def _simulate_without_jit(
    state: PoolState,
    initial_sqrt: int,
    initial_tick: int,
    initial_active_liq: int,
    direction_up: bool,
    net_input: int,
    jit: JITSandwich,
    tx_hash: str,
    dec0: int,
    dec1: int,
    p0: float,
    p1: float,
    fee_rate: float,
) -> tuple[int, list[dict]]:
    """
    Read-only simulation of the swap with JIT liquidity removed.

    Returns (counterfactual_final_sqrtPriceX96, segment_records).

    initial_active_liq must be the pre-walk active liquidity (before _walk_segments
    called state.cross_tick_up/down), because the walk mutates state.active_liq.

    net_input is the estimated net (post-fee) amount in raw token units,
    derived from total_seg_vol so it is consistent with the segment walk's
    estimated liquidity.
    """
    if initial_sqrt == 0:
        return 0, []

    remaining = net_input
    if remaining <= 0:
        return initial_sqrt, []

    # Starting passive liquidity: remove JIT from initial_active_liq if in range.
    # initial_active_liq is the pre-walk state (before tick crossings mutated state.active_liq).
    if jit.tick_lower <= initial_tick < jit.tick_upper:
        current_liq = max(0, initial_active_liq - jit.jit_liquidity)
    else:
        current_liq = initial_active_liq

    current_sqrt = initial_sqrt
    current_tick = initial_tick
    dec_in = dec1 if direction_up else dec0
    p_in = p1 if direction_up else p0

    # Collect all initialized boundaries in the travel direction (no distance cap —
    # the simulation stops when remaining input is exhausted or L hits zero).
    if direction_up:
        boundaries = list(state._sorted_ticks.irange(initial_tick + 1, 887272))
    else:
        boundaries = list(
            state._sorted_ticks.irange(-887272, initial_tick - 1, reverse=True)
        )

    last_sqrt = current_sqrt
    seg_index = 0
    segs: list[dict] = []

    def _make_seg(end_sqrt: int, end_tick: int, consumed_raw: int) -> dict:
        consumed_usd = (consumed_raw / 10**dec_in) * p_in
        return {
            "transaction_hash": tx_hash,
            "segment_index": seg_index,
            "tick_start": current_tick,
            "tick_end": end_tick,
            "sqrt_price_start": str(current_sqrt),
            "sqrt_price_end": str(end_sqrt),
            "price_start_usd": sqrt_x96_to_price(current_sqrt, dec0, dec1),
            "price_end_usd": sqrt_x96_to_price(end_sqrt, dec0, dec1),
            "total_liquidity": current_liq,
            "jit_liquidity": 0,
            "passive_liquidity": current_liq,
            "segment_volume_usd": consumed_usd,
            "fees_total": consumed_usd * fee_rate,
            "fees_to_jit": 0.0,
            "fees_to_passive": consumed_usd * fee_rate,
            "is_counterfactual": True,
        }

    for boundary_tick in boundaries:
        boundary_sqrt = tick_to_sqrt_x96(boundary_tick)
        if boundary_sqrt == 0:
            continue

        if current_liq == 0:
            # No passive depth — advance without consuming input; no segment emitted.
            current_sqrt = boundary_sqrt
            last_sqrt = current_sqrt
            current_tick = boundary_tick
            delta = _jit_adjusted_delta(boundary_tick, state, jit)
            current_liq = current_liq + delta if direction_up else current_liq - delta
            current_liq = max(0, current_liq)
            continue

        if direction_up:
            seg_capacity = current_liq * (boundary_sqrt - current_sqrt) // Q96
            if remaining <= seg_capacity:
                new_sqrt = current_sqrt + remaining * Q96 // current_liq
                new_tick = math.floor(2 * math.log(new_sqrt / Q96) / math.log(1.0001))
                segs.append(_make_seg(new_sqrt, new_tick, remaining))
                return new_sqrt, segs
            segs.append(_make_seg(boundary_sqrt, boundary_tick, seg_capacity))
            seg_index += 1
            remaining -= seg_capacity
            current_sqrt = boundary_sqrt
            last_sqrt = current_sqrt
            current_tick = boundary_tick
            current_liq += _jit_adjusted_delta(boundary_tick, state, jit)
        else:
            if boundary_sqrt >= current_sqrt:
                continue
            seg_capacity = (
                current_liq
                * (current_sqrt - boundary_sqrt)
                * Q96
                // (boundary_sqrt * current_sqrt)
            )
            if remaining <= seg_capacity:
                numerator = current_liq * Q96 * current_sqrt
                denominator = remaining * current_sqrt + current_liq * Q96
                new_sqrt = numerator // denominator
                new_tick = math.floor(2 * math.log(new_sqrt / Q96) / math.log(1.0001))
                segs.append(_make_seg(new_sqrt, new_tick, remaining))
                return new_sqrt, segs
            segs.append(_make_seg(boundary_sqrt, boundary_tick, seg_capacity))
            seg_index += 1
            remaining -= seg_capacity
            current_sqrt = boundary_sqrt
            last_sqrt = current_sqrt
            current_tick = boundary_tick
            current_liq -= _jit_adjusted_delta(boundary_tick, state, jit)

        current_liq = max(0, current_liq)

    # Input exhausted beyond all tracked boundaries; return the furthest sqrt reached.
    return last_sqrt, segs


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def enrich_pool(
    df: pl.DataFrame, cfg: PoolConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
        pl.from_dicts(swap_records, infer_schema_length=None)
        if swap_records
        else pl.DataFrame(),
        pl.from_dicts(segment_records, infer_schema_length=None)
        if segment_records
        else pl.DataFrame(),
        pl.from_dicts(jit_records, infer_schema_length=None)
        if jit_records
        else pl.DataFrame(),
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
