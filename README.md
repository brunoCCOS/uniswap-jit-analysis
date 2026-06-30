# Uniswap V3 JIT Liquidity Analysis Pipeline

Enriches raw Uniswap V3 pool event CSVs with JIT sandwich detection and per-tick fee attribution. Produces Parquet files that quantify how much of each swap's fees went to just-in-time liquidity providers versus passive LPs.

## What it does

For each pool, the pipeline:

1. Loads and sorts all on-chain events (`initialize`, `mint`, `burn`, `swap`) by block → tx index → log index
2. Detects **JIT sandwich attacks** — mint → swap(s) → burn within the same block by the same wallet on the same tick range
3. Reconstructs the pool's tick-level liquidity state event-by-event using V3's sparse tick delta map
4. For each swap, walks every crossed tick segment using V3's constant-liquidity formulas to compute the JIT vs passive liquidity split at each price step
5. Scales segment-level fee estimates to match the actual reported swap volume (correcting for float approximation in intermediate tick sqrt prices)
6. Writes three Parquet files per pool: enriched swaps, per-segment breakdowns, and JIT sandwich summaries

## Repository layout

```
src/
  config.py      — Pool registry (addresses, decimals, fee tiers); set DATA_ROOT here
  loader.py      — CSV ingestion with explicit Polars schema
  state.py       — PoolState: tick delta map, active liquidity, tick crossing helpers
  detector.py    — JIT sandwich detection within a single block
  price.py       — sqrtPriceX96 ↔ price conversions; V3 segment amount formulas
  enricher.py    — Main pipeline: per-block processing, segment walk, fee attribution
scripts/
  process.py     — CLI entry point
tests/
  test_price.py      — Tick math, price conversion, segment amount formulas
  test_state.py      — Pool state machine: mint/burn/cross-tick
  test_detector.py   — JIT detection logic
  test_enricher.py   — Swap handling, fee attribution, active_liq_start timing
output/
  {pool_id}/
    swaps_enriched.parquet
    swap_tick_segments.parquet
    jit_sandwiches.parquet
    metadata.json
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Set `DATA_ROOT` in `src/config.py` to the directory containing the pool subdirectories (e.g. `2697600-eth-usdc-fee-30/`). Each subdirectory must contain a `{pool_id}-Total.csv` file.

## Running

```bash
# Process one pool
python -m scripts.process 2697600

# Process all pools sequentially
python -m scripts.process --all

# Process all pools in parallel (one worker per pool)
python -m scripts.process --all --parallel
```

Produces `output/{pool_id}/` with three Parquet files and a `metadata.json` summary.

## Running tests

```bash
pytest
```

115 tests across price math, state machine, JIT detection, and enrichment logic.

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

## Price and fee math

**sqrtPriceX96 → human price** (token0 per token1, decimal-adjusted):

```
price = 10^(token1_decimals − token0_decimals) / (sqrtPriceX96 / 2^96)^2
```

For USDC(6)/WETH(18): returns USDC per WETH (~2000–4000 range).

**Intermediate tick sqrt prices** are computed with 50-digit decimal precision to avoid float error at large tick values.

**Segment amounts** (V3 constant-liquidity formulas):

```
amount0 = L × |√P_b − √P_a| × 2^96 / (√P_a × √P_b)
amount1 = L × |√P_b − √P_a| / 2^96
```

**Fee attribution** per segment:

```
fees_to_jit     = (jit_liquidity / total_liquidity) × fee_rate × segment_volume_usd
fees_to_passive = total_fees − fees_to_jit
```

Total fees are computed from actual swap volume, then the per-segment JIT share is scaled proportionally so the sum always equals `volume_usd × fee_rate`.

## Output schema

### `swaps_enriched.parquet` — one row per swap

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | str | Swap tx hash |
| `block_number`, `timestamp` | int, str | Block info |
| `sender_address`, `recipient_address`, `txFrom` | str | Swap addresses |
| `amount0`, `amount1` | float | Raw token amounts (positive = pool receives) |
| `token0_price_usd`, `token1_price_usd` | float | Spot prices from CSV |
| `volume_usd` | float | Input-side volume in USD |
| `initial_sqrt_x96`, `final_sqrt_x96` | str | sqrtPriceX96 before and after swap |
| `initial_price`, `final_price` | float | Human-readable token0-per-token1 price |
| `price_impact_pct` | float | (final − initial) / initial × 100 |
| `initial_tick`, `final_tick` | int | Pool tick before and after |
| `ticks_crossed` | int | Number of tick segments traversed |
| `direction` | str | `"buy"` (sqrtPrice up) or `"sell"` (sqrtPrice down) |
| `active_liq_start` | int | Active liquidity at swap start (pre-walk) |
| `active_liq_end` | int | Active liquidity at swap end (on-chain reported) |
| `jit_liquidity_weighted` | float | Volume-weighted avg JIT liquidity across segments |
| `passive_liquidity_weighted` | float | Volume-weighted avg passive liquidity |
| `jit_fraction_weighted` | float | JIT share of active liquidity (volume-weighted) |
| `is_jit` | bool | True if sandwiched by a JIT position |
| `jit_type` | str | `"full"`, `"partial"`, or null |
| `total_fees_usd` | float | Total fees generated by this swap |
| `fees_to_jit_usd` | float | Fees captured by the JIT LP |
| `fees_to_passive_usd` | float | Fees captured by passive LPs |
| `jit_owner` | str | JIT LP wallet address (if `is_jit`) |
| `jit_mint_tx`, `jit_burn_tx` | str | JIT sandwich tx hashes |
| `jit_tick_lower`, `jit_tick_upper` | int | JIT position range |

### `swap_tick_segments.parquet` — one row per tick segment per swap

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | str | FK → swaps_enriched |
| `segment_index` | int | 0-based index within this swap |
| `tick_start`, `tick_end` | int | Tick boundaries of this segment |
| `sqrt_price_start`, `sqrt_price_end` | str | sqrtPriceX96 at each boundary |
| `total_liquidity` | int | Active liquidity during this segment |
| `jit_liquidity` | int | JIT portion of active liquidity |
| `passive_liquidity` | int | Passive portion of active liquidity |
| `segment_volume_usd` | float | Volume attributed to this segment |
| `fees_total` | float | Total fees for this segment |
| `fees_to_jit` | float | JIT fee share |
| `fees_to_passive` | float | Passive fee share |

### `jit_sandwiches.parquet` — one row per detected JIT sandwich

| Column | Type | Description |
|---|---|---|
| `sandwich_id` | str | Unique ID (= mint tx hash) |
| `block_number`, `timestamp` | int, str | Block info |
| `owner` | str | JIT LP wallet address |
| `mint_tx`, `burn_tx` | str | Sandwich transaction hashes |
| `tick_lower`, `tick_upper` | int | Position range |
| `jit_liquidity` | int | Liquidity minted |
| `burn_liquidity` | int | Liquidity burned (may differ for partial JIT) |
| `jit_type` | str | `"full"` or `"partial"` |
| `new_passive_liq` | int | Liquidity that stayed after partial burn |
| `swap_count` | int | Number of sandwiched swaps |
| `total_volume_usd` | float | Aggregate swap volume |
| `fees_captured_usd` | float | Fees earned by the JIT position |
| `total_fees_usd` | float | Total fees generated by sandwiched swaps |

## Pool results

| Pool ID | Pair | Fee | Events | Swaps | JIT sandwiches | Runtime |
|---|---|---|---|---|---|---|
| 2697585 | USDC/WETH | 1.00% | 9,127 | 5,090 | 6 | 0.2s |
| 2697588 | USDC/USDT | 0.05% | 20,698 | 8,369 | 32 | 0.3s |
| 2697600 | USDC/WETH | 0.30% | 247,478 | 65,536 | 615 | 7.1s |
| 2697647 | WBTC/USDC | 0.30% | 43,173 | 15,498 | 168 | 0.8s |
| 2697765 | USDC/WETH | 0.05% | 1,428,341 | 1,180,655 | 8,483 | 916.7s |

JIT rate is highest on the 0.05% WETH pool (2697765): 8,483 sandwiches across 1.18M swaps (~0.7% of swaps sandwiched). The 1% WETH pool (2697585) has only 6 JIT events, consistent with low sandwich profitability at wider spreads.

## Known limitations

- **Greedy JIT matching**: when a wallet mints twice in one block on the same tick range, the earliest valid burn is matched to the first mint. The second mint is unmatched even if a later burn exists.
- **Hard-coded data root**: `DATA_ROOT` in `config.py` must be updated manually; it is not a CLI argument.
- **Float precision in prices**: `initial_price`/`final_price` columns use Python float, which loses ~4 ULP at large sqrtPriceX96 values. This does not affect fee calculations (those use on-chain USD prices from the CSV).
- **No test data fixtures**: the `tests/` directory contains unit tests only; integration tests against real pool data are not included.
