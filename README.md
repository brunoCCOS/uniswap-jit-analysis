# Uniswap V3 JIT Liquidity Analysis Pipeline

Enriches raw Uniswap V3 pool event CSVs with JIT sandwich detection and per-tick fee attribution.

## What it does

For each pool, the pipeline:
1. Loads and sorts all events (`mint`, `burn`, `swap`, etc.)
2. Detects **JIT sandwich attacks** — mint → swap(s) → burn in the same block by the same wallet
3. Reconstructs the pool's tick-level liquidity state event-by-event
4. For each swap, walks every crossed tick segment to compute how much liquidity (JIT vs passive) was active at each point
5. Outputs enriched Parquet files with per-swap metrics and per-tick-segment breakdowns

## JIT detection logic

A JIT sandwich requires:
- Mint and burn from the **same `owner_address`** in the same block
- Burn has a **higher transaction index** than the mint
- At least one swap occurs **between** mint and burn (by tx index)
- Same `tickLower`/`tickUpper` on mint and burn (same position range)

Types:
- **Full JIT**: `burn_liquidity == mint_liquidity` (all injected liquidity removed)
- **Partial JIT**: `burn_liquidity < mint_liquidity` (some liquidity stays as passive)

## Fee attribution

For each swap, the pipeline walks every tick boundary crossed using Uniswap V3's exact formulas:

```
segment_amount0 = L × (√P_a - √P_b) × Q96 / (√P_a × √P_b)   # token0 input
segment_amount1 = L × (√P_b - √P_a) / Q96                      # token1 input
```

At each segment:
- JIT liquidity = sum of active JIT positions whose range covers the current tick
- Passive liquidity = total_active_liquidity − JIT_liquidity
- fees_to_jit = (jit_liq / total_liq) × fee_rate × segment_volume_usd

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Configure pool data paths in `src/config.py` (`DATA_ROOT`).

## Usage

```bash
# Process one pool
python -m scripts.process 2697600

# Process all pools sequentially
python -m scripts.process --all

# Process all pools in parallel
python -m scripts.process --all --parallel
```

## Output

Each pool produces three files in `output/{pool_id}/`:

### `swaps_enriched.parquet`
One row per swap event.

| Column | Description |
|---|---|
| `transaction_hash` | Swap tx hash |
| `block_number`, `timestamp` | Block info |
| `initial_tick`, `final_tick` | Pool tick before and after swap |
| `initial_price`, `final_price` | Human-readable price (token1/token0, decimal-adjusted) |
| `price_impact_pct` | (final − initial) / initial × 100 |
| `ticks_crossed` | Number of tick boundaries traversed |
| `active_liq_start` | Pool active liquidity at swap start |
| `jit_liquidity_weighted` | Volume-weighted avg JIT liquidity across segments |
| `passive_liquidity_weighted` | Volume-weighted avg passive liquidity |
| `jit_fraction_weighted` | JIT share of active liquidity (volume-weighted) |
| `is_jit` | True if this swap is part of a JIT sandwich |
| `jit_type` | `"full"`, `"partial"`, or null |
| `volume_usd` | Input-side swap volume in USD |
| `total_fees_usd` | Total fees generated |
| `fees_to_jit_usd` | Fees captured by the JIT LP |
| `fees_to_passive_usd` | Fees captured by passive LPs |
| `jit_owner`, `jit_mint_tx`, `jit_burn_tx` | JIT sandwich metadata |

### `swap_tick_segments.parquet`
One row per tick segment per swap (for multi-tick swaps). Enables full state reconstruction.

| Column | Description |
|---|---|
| `transaction_hash` | FK to swaps_enriched |
| `segment_index` | 0-based index within swap |
| `tick_start`, `tick_end` | Tick range of this segment |
| `total_liquidity`, `jit_liquidity`, `passive_liquidity` | Liquidity at each segment |
| `segment_volume_usd` | Volume attributed to this segment |
| `fees_to_jit`, `fees_to_passive` | Fee split for this segment |

### `jit_sandwiches.parquet`
One row per detected JIT sandwich.

| Column | Description |
|---|---|
| `sandwich_id` | Unique ID (mint tx hash) |
| `owner` | JIT LP wallet address |
| `jit_liquidity`, `burn_liquidity` | Liquidity minted and burned |
| `jit_type` | `"full"` or `"partial"` |
| `swap_count` | Number of swaps sandwiched |
| `total_volume_usd`, `fees_captured_usd` | Aggregate swap metrics |

## Pools

| Pool ID | Pair | Fee | Events | Swaps | JIT count |
|---|---|---|---|---|---|
| 2697585 | USDC/WETH | 100bp | 9,127 | 5,090 | 6 |
| 2697588 | USDC/USDT | 5bp | 20,698 | 8,369 | 32 |
| 2697600 | USDC/WETH | 30bp | 247,478 | 65,536 | 615 |
| 2697647 | WBTC/USDC | 30bp | 43,173 | 15,498 | 168 |
| 2697765 | USDC/WETH | 5bp | 1,428,341 | 1,180,655 | 8,483 |
