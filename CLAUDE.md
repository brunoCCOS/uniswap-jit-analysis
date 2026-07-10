# Ground Truths for the JIT Analysis Pipeline

## How the pipeline is structured

The enricher (`src/enricher.py`) walks all pool events in chronological order, maintaining a `PoolState` that tracks:
- `tick_deltas`: sparse map of tick → net liquidity delta, updated by every mint/burn
- `active_liq`: the pool's active liquidity at the current tick, updated by `apply_swap` using the on-chain reported value (ground truth), and updated tick-by-tick via `cross_tick_up/down` during the segment walk
- `sqrt_x96`: current price in Q64.96 format, updated only by `apply_swap`
- `tick`: current tick, updated only by `apply_swap`

**Critical**: `cross_tick_up/down` (called during `_walk_segments`) mutates `active_liq` but NOT `tick` or `sqrt_x96`. Those are only updated at the end of `_handle_swap` via `apply_swap`. So when the optimizers are called (between `_walk_segments` and `apply_swap`), `state.tick` and `state.sqrt_x96` are still the INITIAL values for the swap — NOT the final ones.

## Passive liquidity reconstruction

`_build_passive_dict` reconstructs the passive liquidity profile from:
1. `active_liq_start`: the pre-walk active liquidity (includes JIT; captured before `_walk_segments` mutates it)
2. `tick_deltas`: all mint/burn boundaries accumulated so far

The base passive liq at `start_lower = (tick // ts) * ts` is:
```
base_liq = active_liq_start - jit.jit_liquidity  (if JIT covers current tick)
         = active_liq_start                       (otherwise)
```

This is walked up and down through `tick_deltas` (with JIT's contribution removed via `_passive_delta`) to produce the full profile, then densified so every tick-spacing multiple has an entry.

**The CSV starts from pool genesis (pool creation block).** Every mint and burn ever executed on the pool is in the CSV. Therefore `tick_deltas` contains every initialized tick boundary — no historical positions are missing. The passive dict is correctly reconstructed.

## Unit scaling (library vs raw Uniswap)

The JITUniswapOptimization library uses human-unit sqrt prices and liquidity:
- `sqrt_human = sqrt_x96 / Q96 * 10^((dec0 - dec1) / 2)`
- `L_lib = L_raw / liq_scale` where `liq_scale = 10^((dec0 + dec1) / 2)`
- `amount_in_lib = amount_in_raw / 10^dec_in`

For USDC(6)/WETH(18): `liq_scale = 10^12`.
For USDC(6)/USDT(6): `liq_scale = 10^6`.

## The two simulators

### Enricher's `_simulate_without_jit` (Python integer arithmetic, Q96)
- Used to compute the counterfactual "no-JIT final tick"
- Uses `initial_active_liq` (pre-walk, raw Uniswap L) as starting passive liquidity
- Walks through `state._sorted_ticks` (all tracked boundaries)
- Stops when `remaining` is exhausted or all boundaries processed
- **No tick-distance cap** — walks as far as needed

### JITUniswapOptimization `Swap.simulate()` (Q96 integer arithmetic)
- Used by both optimizers to score candidate JIT positions
- Reads passive liquidity from `passive_dict` (the dense dict from `_build_passive_dict`)
- Internally converts lib-unit inputs to raw Q96 integers, runs the hot loop in pure integer arithmetic, then converts outputs back to lib units — external interface unchanged
- **Three caches on the Swap instance** (valid for the lifetime of a Swap object, which is one JIT event):
  - `_passive_raw`: lib→raw conversion of passive_dict (reused across all ~700 simulate() calls per JIT event)
  - `_bsqrt_cache`: tick→raw-sqrt-x96 lookups (float pow computed once per boundary tick, then dict lookup)
  - `_sim_cache_id`: tracks `id(state)` to invalidate the above if state changes
- `output_amount` is always returned as `0.0` — the optimizer only needs `fees_jit_lp` and `final_sqrt_price`, and computing output required expensive 200-bit BigInt multiplications
- Regression-tested against the original Decimal implementation to within 0.05% on fees and final price

## Degenerate pool case (pool 2697588 — USDC/USDT ts=10)

Some swaps in pool 2697588 have a JIT position that owns ~99.9957% of the pool's active liquidity. After subtracting JIT, passive liquidity ≈ 8939 lib units (~8.9e9 raw). A swap of ~35,000 USDT at this thin passive depth moves the price by ~80,000 ticks (USDT "price" going from ~$1.03 to impossible levels). This is correct — WITHOUT the JIT, the pool is essentially illiquid and the swap would cause catastrophic price impact.

The Q96 simulator handles this correctly at ~3.3ms per simulate() call (warm cache), vs the old Decimal approach which stalled at ~530ms+. 700 calls per JIT event × 3.3ms ≈ 2.3 seconds per JIT event.

## Key invariants

1. `active_liq_start` is always the correct on-chain active liq at the swap's starting tick (captured pre-walk, anchored by the previous `apply_swap` ground-truth value).
2. The passive dict at `start_lower` equals `active_liq_start - jit_liq` (if current tick is inside JIT range) or `active_liq_start` (otherwise).
3. Above JIT's upper boundary, passive liq equals the pre-JIT passive liq (JIT delta cancels out in `_passive_delta`).
4. `no_jit_final_tick` from the enricher and the optimizer's internal no-JIT simulation both use Q96 integer arithmetic now, but may still differ slightly because the enricher walks only initialized tick boundaries (sparse) while the optimizer walks a dense dict (every ts multiple). They should agree for pools with enough initialized boundaries, and diverge only in very thin regions.
5. JIT can only slow the price — with JIT, the final tick is always ≤ no-JIT final tick (for up-swaps) or ≥ (for down-swaps).

## Analytical vs combinatorial optimizers

Both optimizers should converge to the same result because they maximise the same utility function. Differences arise only from:
- Numerical precision (Decimal vs float)
- The combinatorial's `max_number_pos` cap (can miss the true optimum if it's outside the tried positions)
- The analytical assuming containment (utility regime) incorrectly when `L < L0`

The `_range_utility` helper in `analytical.py` correctly handles the fully-crossed (`L < L0`) regime by switching to the linear approximation `u_fc = L * fc_slope`.
