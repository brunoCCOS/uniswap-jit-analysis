# Uniswap V3 JIT Liquidity Analysis Pipeline

Enriches raw Uniswap V3 pool event CSVs with JIT sandwich detection, per-tick fee attribution, counterfactual simulations, and optimal JIT allocation search. Produces Parquet files that let you quantify how much of each swap's fees went to just-in-time liquidity providers versus passive LPs — and how much value the JIT bot left on the table.

## What it does

For each pool, the pipeline:

1. Loads and sorts all on-chain events (`initialize`, `mint`, `burn`, `swap`) by block → tx index → log index
2. Detects **JIT sandwich attacks** — mint → swap(s) → burn within the same block by the same wallet on the same tick range
3. Reconstructs the pool's tick-level liquidity state event-by-event using V3's sparse tick delta map
4. For each swap, walks every crossed tick segment using V3's constant-liquidity formulas to compute the JIT vs passive liquidity split at each price step
5. Scales segment-level fee estimates to match actual reported swap volume (correcting for float approximation in intermediate tick sqrt prices)
6. For each JIT swap: re-simulates the swap **without** the JIT position to produce counterfactual tick segments and final price
7. For each JIT swap: searches ±2 tick-spacings around the actual JIT range to find the tick range and liquidity level that would have maximised the JIT LP's utility (fees + mark-to-market P&L) given the same capital
8. Writes three Parquet files per pool: enriched swaps, per-segment breakdowns, and JIT sandwich summaries

---

## Repository layout

```
src/
  config.py        — Pool registry (addresses, decimals, fee tiers); set DATA_ROOT here
  loader.py        — CSV ingestion with explicit Polars schema
  state.py         — PoolState: tick delta map, active liquidity, tick crossing helpers
  detector.py      — JIT sandwich detection within a single block
  price.py         — sqrtPriceX96 ↔ price conversions; V3 segment amount formulas
  enricher.py      — Main pipeline: per-block processing, segment walk, fee attribution,
                     counterfactual simulation, optimizer call
  jit_optimizer.py — Wrapper around JITUniswapOptimization library: builds passive liquidity
                     dict, coordinate conversions, ternary-search optimisation
scripts/
  process.py            — CLI entry point (single pool or --all [--parallel])
  analyze_tick_crossings.py — Distribution of ticks crossed per swap (--no-jit for counterfactual)
tests/
  test_price.py         — Tick math, price conversion, segment amount formulas
  test_state.py         — Pool state machine: mint/burn/cross-tick
  test_detector.py      — JIT detection logic
  test_enricher.py      — Swap handling, fee attribution, active_liq_start timing
  test_integration.py   — End-to-end pipeline checks on real pool data
output/
  {pool_id}/
    swaps_enriched.parquet       — one row per swap (all swaps)
    swap_tick_segments.parquet   — one row per tick segment per swap (actual + counterfactual)
    jit_sandwiches.parquet       — one row per detected JIT sandwich
    metadata.json                — pool summary stats
```

External dependency: `/home/brunollacer/JITUniswapOptimization` — a Uniswap V3 JIT optimisation
library (float/Decimal model). Imported via `sys.path.insert` in `jit_optimizer.py`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install numpy  # required by JITUniswapOptimization
```

Set `DATA_ROOT` in `src/config.py` to the directory containing pool subdirectories
(e.g. `2697600-eth-usdc-fee-30/`). Each subdirectory must contain a `{pool_id}-Total.csv` file.

## Running

```bash
# Process one pool
python -m scripts.process 2697600

# Process all pools sequentially
python -m scripts.process --all

# Process all pools in parallel (one worker per pool)
python -m scripts.process --all --parallel

# Tick-crossing distribution (actual)
python -m scripts.analyze_tick_crossings

# Tick-crossing distribution (counterfactual — no JIT)
python -m scripts.analyze_tick_crossings --no-jit
```

## Running tests

```bash
pytest        # 158 tests
```

---

## Coordinate systems and conversions

Understanding the pipeline requires knowing three coordinate systems and how they relate.

### 1. On-chain: sqrtPriceX96 and raw liquidity

Uniswap V3 stores price as a Q64.96 fixed-point integer:

```
sqrtPriceX96 = sqrt(token1_raw / token0_raw) × 2^96
```

where `token1_raw / token0_raw` is the ratio of undivided (raw) token amounts. No decimal adjustment.

Active liquidity `L_raw` is also an on-chain integer. Token amounts moved through a constant-L segment are:

```
Δtoken0_raw = L_raw × (1/√P_a − 1/√P_b) × 2^96 / 2^96        [simplified]
Δtoken1_raw = L_raw × (√P_b − √P_a) / 2^96
```

All state tracking in `state.py`, `enricher.py`, `price.py` uses this coordinate system exclusively.

### 2. Human-readable: decimal-adjusted price

To show a price that means something (e.g. USDC per WETH):

```
price = 10^(dec1 − dec0) / (sqrtPriceX96 / 2^96)^2
```

For USDC(dec0=6)/WETH(dec1=18): `10^12 / (sqrtP/2^96)^2` gives USDC per WETH (~2000–4000).

Tick↔price:

```
sqrtPriceX96 / 2^96  =  1.0001^(tick/2)
tick                 =  floor( 2 × log(sqrtPriceX96 / 2^96) / log(1.0001) )
```

Implemented in `price.py` as `sqrt_x96_to_price` and used for all `*_price` and `*_price_usd` columns.

### 3. Library (JITUniswapOptimization): adjusted sqrt price and scaled liquidity

The optimiser library uses a **decimal-adjusted** sqrt price and a **scaled liquidity**:

```
sqrt_lib  = sqrtPriceX96 / 2^96 / 10^((dec1−dec0)/2)
          = 1.0001^(tick/2) / 10^((dec1−dec0)/2)

L_lib     = L_raw / 10^((dec0+dec1)/2)
```

This pairing ensures the standard V3 formulas produce **human-readable token amounts** directly:

```
Δtoken0_human = L_lib × (1/sqrt_lib_lower − 1/sqrt_lib)
Δtoken1_human = L_lib × (sqrt_lib − sqrt_lib_lower)
```

Passing `L_raw` without scaling would inflate amounts by `10^((dec0+dec1)/2)`. For USDC/WETH that is `10^12` — making tick-crossing comparisons nonsensical and utility calculations meaningless.

**Conversion in `jit_optimizer.py`:**

```python
dec_scale = 10 ** ((dec0 + dec1) / 2)
sqrt_lib  = Decimal(str(initial_sqrt / 2**96 / 10**((dec1-dec0)/2)))
L_lib     = float(L_raw) / dec_scale
```

Stored `optimal_jit_liquidity` is converted back to raw units (`L_lib × dec_scale`) so it is directly comparable to the on-chain `jit_liquidity` values in `jit_sandwiches.parquet`.

---

## JIT detection logic

A JIT sandwich requires, within a single block:

- Mint and burn from the **same `owner_address`**
- Burn at a **higher transaction index** than the mint
- At least one swap with a transaction index **strictly between** mint and burn
- **Same `tickLower`/`tickUpper`** on mint and burn

**JIT type:**
- `full` — `burn_liquidity >= mint_liquidity` (all injected liquidity removed)
- `partial` — `burn_liquidity < mint_liquidity` (some stays as passive; tracked in `new_passive_liq`)

Matching is greedy: the first valid burn after each mint is used. A burn can only be credited to one mint.

---

## Price and fee math

### sqrtPriceX96 → human price

```
price = 10^(dec1 − dec0) / (sqrtPriceX96 / 2^96)^2
```

Intermediate tick sqrt prices used in the segment walk are computed with 50-digit `Decimal` precision to avoid float error accumulation at large tick values.

### Segment amount formulas (V3 constant-L)

For a segment from sqrt price `P_a` to `P_b` with active liquidity `L` (all in raw/Q96 units):

```
amount0 = L × |1/√P_a − 1/√P_b| × 2^96
amount1 = L × |√P_b − √P_a| / 2^96
```

The segment whose direction matches the swap direction contributes to `segment_volume_usd`.

### Fee attribution per segment

```
fees_total       = segment_volume_usd × fee_rate
fees_to_jit      = (jit_liquidity / total_liquidity) × fees_total
fees_to_passive  = fees_total − fees_to_jit
```

Because intermediate sqrt prices are approximations, segment volumes are re-scaled so their sum exactly matches the on-chain `volume_usd`:

```
scale = volume_usd / sum(segment_volume_usd for all segments)
each segment_volume_usd *= scale
fees scaled proportionally
```

### JIT position USD value (`jit_liquidity_usd`)

Computed in `price.py::position_value_usd` using Q64.96 integer arithmetic:

```
amount0, amount1 = V3 formulas(L_raw, sqrtP_lower, sqrtP_upper, sqrtP_current)
jit_liquidity_usd = amount0 × price0_usd + amount1 × price1_usd
```

This value is stored for reference but is **not** used as the optimiser budget. See Optimiser section.

### Counterfactual price impact

For JIT swaps, `_simulate_without_jit` replays the swap using only passive liquidity (JIT delta removed from all tick boundaries). The result:

```
no_jit_price_impact_pct = (no_jit_final_price − initial_price) / initial_price × 100
```

A larger absolute value than `price_impact_pct` means the JIT absorbed price impact that would otherwise have been borne by the swap.

---

## JIT optimiser

For each JIT swap, the pipeline searches for the tick range and liquidity amount that would have maximised the JIT LP's **utility** given the same capital the actual bot deployed.

### Utility definition

```
utility = (end_value − start_value) + fees_earned
        = mark-to-market P&L + fees
```

where `start_value` and `end_value` are the USD value of the JIT position's token holdings before and after the swap, and `fees_earned` is the JIT's proportional share of swap fees.

A negative utility means the impermanent loss exceeded the fees earned.

### Budget derivation

The budget is the USD value of the **actual** JIT position, computed using the library's own float/Decimal model:

```python
actual_pos = Position(L_raw / dec_scale, tick_lower, tick_upper)
budget     = actual_pos.value(sqrt_lib, price0, price1, dec0, dec1)
         # = amount0_human × price0 + amount1_human × price1
```

Using the library's model (not our Q64.96 `position_value_usd`) ensures the round-trip is exact:

```
liquidity_from_budget(budget, same_range) → L_lib ≈ L_raw / dec_scale
```

Both forward and inverse use the same float/Decimal arithmetic so no cross-model scale mismatch.

### Search procedure

1. Build `passive_dict = {rounded_tick: L_lib}` covering every `tick_spacing` step from `initial_tick` to `no_jit_final_tick`. At each step the passive liquidity is the last known value (carry-forward from initialized boundaries), with JIT delta removed.
2. For each candidate range `[a, b]` in `{actual_lower ± 2×tick_spacing} × {actual_upper ± 2×tick_spacing}`:
   a. Compute `max_liq = liquidity_from_budget(budget, [a, b])` — maximum allocatable L_lib
   b. Run ternary search on `utility(L) for L in [0, max_liq]` to find the optimal liquidity level
3. Keep the `[a, b, L]` triple with the highest utility.
4. Compare against `utility(actual_pos)` at the actual on-chain range and liquidity.

Ternary search epsilon: `max(max_liq × 1e-3, 1.0)` — relative tolerance that keeps iteration count bounded at ~25 regardless of liquidity scale.

All library stdout (including "Liquidity is zero" debug prints) is suppressed via `contextlib.redirect_stdout`.

---

## Output schema

### `swaps_enriched.parquet` — one row per swap

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | str | Swap tx hash |
| `block_number`, `timestamp` | int, str | Block info |
| `sender_address`, `recipient_address`, `txFrom` | str | Swap addresses |
| `amount0`, `amount1` | float | Raw token amounts (positive = pool receives) |
| `token0_price_usd`, `token1_price_usd` | float | Spot prices from input CSV |
| `volume_usd` | float | Input-side volume in USD |
| `initial_sqrt_x96`, `final_sqrt_x96` | str | sqrtPriceX96 before and after swap (stored as string to preserve precision) |
| `initial_price`, `final_price` | float | Human-readable token0-per-token1 price (see coordinate system 2) |
| `price_impact_pct` | float | `(final − initial) / initial × 100` |
| `initial_tick`, `final_tick` | int | On-chain pool tick before and after |
| `ticks_crossed` | int | Number of tick boundaries traversed by the actual swap |
| `direction` | str | `"buy"` (sqrtPrice rises, token1→token0) or `"sell"` (sqrtPrice falls, token0→token1) |
| `active_liq_start` | int | `L_raw` at swap start, **after** JIT mint is applied (on-chain state) |
| `active_liq_end` | int | `L_raw` at swap end (on-chain reported) |
| `jit_liquidity_weighted` | float | Volume-weighted average JIT liquidity across segments |
| `passive_liquidity_weighted` | float | Volume-weighted average passive liquidity |
| `jit_fraction_weighted` | float | JIT share of active liquidity (volume-weighted); 0 for non-JIT swaps |
| `is_jit` | bool | True if this swap is sandwiched by a JIT position |
| `jit_type` | str | `"full"`, `"partial"`, or null |
| `total_fees_usd` | float | Total fees generated: `volume_usd × fee_rate` |
| `fees_to_jit_usd` | float | Fees captured by the JIT LP (0 for non-JIT swaps) |
| `fees_to_passive_usd` | float | Fees captured by passive LPs |
| `jit_owner` | str | JIT LP wallet address (null if not JIT) |
| `jit_mint_tx`, `jit_burn_tx` | str | JIT sandwich transaction hashes |
| `jit_tick_lower`, `jit_tick_upper` | int | Actual on-chain JIT position boundaries |
| `jit_liquidity_usd` | float | USD TVL of the JIT position at swap start; computed with Q64.96 math (null if not JIT) |
| `initial_tick_price`, `final_tick_price` | float | Price at the rounded tick boundary (differs from `initial_price`/`final_price` which use actual sqrtPriceX96) |
| `no_jit_final_sqrt_x96` | str | Counterfactual final sqrtPriceX96 if JIT had not provided liquidity (null if not JIT) |
| `no_jit_final_tick` | int | Counterfactual final tick without JIT |
| `no_jit_final_price` | float | Counterfactual final price without JIT |
| `no_jit_price_impact_pct` | float | Counterfactual price impact; larger magnitude than `price_impact_pct` when JIT absorbed price impact |
| `optimal_tick_lower`, `optimal_tick_upper` | int | Best tick range found by the optimiser (null if not JIT) |
| `optimal_jit_liquidity` | float | Optimal liquidity at that range, in raw on-chain units (null if not JIT) |
| `optimal_utility_usd` | float | Utility (P&L + fees) at the optimal allocation (null if not JIT) |
| `actual_utility_usd` | float | Utility of the actual on-chain JIT position (null if not JIT); negative means the bot lost money on this swap |

**Interpretation notes:**
- `optimal_utility_usd >= actual_utility_usd` always holds (optimiser is a maximum search). The gap quantifies how much the JIT bot underperformed relative to the best allocation of the same capital.
- `actual_utility_usd < 0` means impermanent loss exceeded fees on this specific swap.
- `optimal_utility_usd = 0` with `actual_utility_usd < 0` means no tick range in the search neighbourhood could turn a profit; the bot should not have participated.

---

### `swap_tick_segments.parquet` — one row per tick segment per swap

Each swap is decomposed into segments separated by tick boundaries. Both the **actual** walk and the **counterfactual** (no-JIT) walk are stored, distinguished by `is_counterfactual`.

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | str | FK → swaps_enriched |
| `segment_index` | int | 0-based index within this swap's walk (resets for counterfactual) |
| `tick_start`, `tick_end` | int | Tick at start and end of this segment |
| `sqrt_price_start`, `sqrt_price_end` | str | sqrtPriceX96 at each boundary (string for precision) |
| `price_start_usd`, `price_end_usd` | float | Human-readable token0-per-token1 price at each boundary |
| `total_liquidity` | int | Active `L_raw` during this segment |
| `jit_liquidity` | int | JIT portion of active liquidity (0 for counterfactual segments) |
| `passive_liquidity` | int | Passive portion of active liquidity |
| `segment_volume_usd` | float | Volume attributed to this segment (after scaling) |
| `fees_total` | float | Total fees for this segment |
| `fees_to_jit` | float | JIT fee share (0 for counterfactual) |
| `fees_to_passive` | float | Passive fee share |
| `is_counterfactual` | bool | False = actual swap walk; True = counterfactual (no-JIT) walk |

**Reading segments:**

For a JIT swap there are two walks sharing the same `transaction_hash`:
- `is_counterfactual = False`: the actual swap. The JIT's liquidity is present. Typically fewer segments because the JIT concentrates liquidity and absorbs more of the swap within a narrow range.
- `is_counterfactual = True`: the same swap replayed without JIT liquidity. Price moves further (more segments, wider tick spread). Comparing these two walks shows how much price impact the JIT absorbed.

The `price_start_usd` / `price_end_usd` columns express price as **token0 per token1** in the same convention as `initial_price` in `swaps_enriched`. For USDC/WETH pools this is USDC per WETH. For a price-down (sell) swap, `price_start_usd > price_end_usd`.

---

### `jit_sandwiches.parquet` — one row per detected JIT sandwich

| Column | Type | Description |
|---|---|---|
| `sandwich_id` | str | Unique ID (= mint tx hash) |
| `block_number`, `timestamp` | int, str | Block info |
| `owner` | str | JIT LP wallet address |
| `mint_tx`, `burn_tx` | str | Sandwich transaction hashes |
| `tick_lower`, `tick_upper` | int | Position tick range |
| `jit_liquidity` | int | `L_raw` minted |
| `burn_liquidity` | int | `L_raw` burned (may be less for partial JIT) |
| `jit_type` | str | `"full"` or `"partial"` |
| `new_passive_liq` | int | Liquidity retained after partial burn |
| `swap_count` | int | Number of sandwiched swaps |
| `total_volume_usd` | float | Aggregate swap volume of sandwiched swaps |
| `fees_captured_usd` | float | Total fees earned by the JIT position across all sandwiched swaps |
| `total_fees_usd` | float | Total fees generated by those swaps (JIT + passive) |

---

## Pool results (after all fixes)

| Pool ID | Pair | Fee | Events | Swaps | JIT sandwiches | Runtime |
|---|---|---|---|---|---|---|
| 2697585 | USDC/WETH | 1.00% | 9,127 | 5,090 | 6 | 0.5s |
| 2697588 | USDC/USDT | 0.05% | 20,698 | 8,369 | 32 | 1.6s |
| 2697600 | USDC/WETH | 0.30% | 247,478 | 65,536 | 615 | 18s |
| 2697647 | WBTC/USDC | 0.30% | 43,173 | 15,498 | 168 | 2.6s |
| 2697765 | USDC/WETH | 0.05% | 1,428,341 | 1,180,655 | 8,483 | ~12 min |

JIT rate is highest on the 0.05% WETH pool (2697765): 8,483 sandwiches across 1.18M swaps (~0.7% of swaps sandwiched). The 1% WETH pool (2697585) has only 6 JIT events, consistent with low sandwich profitability at wider spreads.

---

## Known limitations

- **Greedy JIT matching**: when a wallet mints twice in one block on the same tick range, the earliest valid burn is matched to the first mint. The second mint is unmatched even if a later burn exists.
- **Optimiser fee formula**: the JITUniswapOptimization library computes fees as `fee = net_in × fee_rate` rather than the V3 spec's `fee = gross_in × fee_rate`, underestimating fees by ~`fee_rate` in relative terms (~0.3% at the 0.3% tier). This does not affect the optimality ranking but slightly deflates both `optimal_utility_usd` and `actual_utility_usd`.
- **Optimiser search radius**: the tick range search is bounded to ±2 tick-spacings around the actual JIT position for runtime tractability. A globally optimal range far from the actual position is not found.
- **Hard-coded data root**: `DATA_ROOT` in `config.py` must be updated manually.
- **Float precision in prices**: `initial_price`/`final_price` columns use Python float (~15 sig figs). This does not affect fee calculations (those use on-chain USD prices from the CSV).
