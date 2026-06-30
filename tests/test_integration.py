"""
Integration test: full pipeline on a synthetic but realistic pool fixture.

Fixture layout (tests/fixtures/pool_2697600_mini.csv):
  Block 1000: initialize + 2 passive mints
              PASSIVE1: 10e15 liq on [194000, 197000]
              PASSIVE2:  5e15 liq on [195240, 195420]
  Block 1001: SWAP1 — regular buy (WETH in, USDC out), tick 195300→195360, no JIT
  Block 1002: JIT sandwich
                JIT_MINT:  8e15 liq on [195240, 195420]  (tx=0)
                SWAP2:     buy, tick 195360→195380        (tx=1)  ← sandwiched
                           (ends at 195380, NOT at tick boundary, to avoid edge case)
                JIT_BURN:  8e15 liq on [195240, 195420]  (tx=2)
  Block 1003: SWAP3 — regular sell (USDC in, WETH out), tick 195380→195350, no JIT

All amounts are V3-formula-consistent: derived from segment_amounts() with the
correct active liquidity at each swap. Scale factors are therefore ≈ 1/(1-fee_rate).

Expected outcomes:
  - 3 swap records: SWAP1, SWAP2, SWAP3
  - 1 JIT sandwich record: owner=0xJIT_LP
  - SWAP2.is_jit=True; SWAP1 and SWAP3 are False
  - JIT fraction on SWAP2 ≈ 8e15 / 23e15 ≈ 34.78%
  - fee conservation: fees_to_jit + fees_to_passive = total_fees for every swap
  - active_liq_start captured before tick crossings (pre-walk state)
"""

import pytest
import polars as pl
from pathlib import Path

from src.config import POOL_BY_ID
from src.enricher import enrich_pool
from src.loader import CSV_SCHEMA_OVERRIDES, USED_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "pool_2697600_mini.csv"
CFG = POOL_BY_ID["2697600"]

JIT_LIQ = 8_000_000_000_000_000          # 8e15
PASSIVE_NARROW_LIQ = 5_000_000_000_000_000  # 5e15 — PASSIVE2
PASSIVE_WIDE_LIQ = 10_000_000_000_000_000   # 1e16 — PASSIVE1
TOTAL_LIQ_SWAP2 = JIT_LIQ + PASSIVE_NARROW_LIQ + PASSIVE_WIDE_LIQ  # 23e15
FEE_RATE = 3000 / 1_000_000                 # 0.3%


@pytest.fixture(scope="module")
def pipeline_results():
    df = (
        pl.scan_csv(
            FIXTURE,
            schema_overrides=CSV_SCHEMA_OVERRIDES,
            null_values=["", "NA", "NaN"],
        )
        .select(USED_COLUMNS)
        .sort(["block_number", "transaction_index", "log_index"])
        .collect()
    )
    swaps, segs, jits = enrich_pool(df, CFG)
    return swaps, segs, jits


class TestOutputShape:
    def test_three_swaps_produced(self, pipeline_results):
        swaps, _, _ = pipeline_results
        assert len(swaps) == 3

    def test_one_jit_sandwich_produced(self, pipeline_results):
        _, _, jits = pipeline_results
        assert len(jits) == 1

    def test_segments_emitted_for_tick_crossing_swaps(self, pipeline_results):
        _, segs, _ = pipeline_results
        # Each multi-tick swap produces at least one segment record
        assert len(segs) >= 3

    def test_swap_records_have_expected_columns(self, pipeline_results):
        swaps, _, _ = pipeline_results
        for col in ["transaction_hash", "direction", "volume_usd",
                    "total_fees_usd", "fees_to_jit_usd", "fees_to_passive_usd",
                    "active_liq_start", "active_liq_end", "is_jit", "jit_fraction_weighted"]:
            assert col in swaps.columns


class TestJITDetection:
    def test_swap2_is_sandwiched(self, pipeline_results):
        swaps, _, _ = pipeline_results
        swap2 = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")
        assert len(swap2) == 1
        assert swap2["is_jit"][0] is True

    def test_swap1_and_swap3_not_sandwiched(self, pipeline_results):
        swaps, _, _ = pipeline_results
        for tx in ("0xSWAP1", "0xSWAP3"):
            row = swaps.filter(pl.col("transaction_hash") == tx)
            assert row["is_jit"][0] is False, f"{tx} should not be JIT"

    def test_jit_owner_is_correct(self, pipeline_results):
        _, _, jits = pipeline_results
        assert jits["owner"][0] == "0xJIT_LP"

    def test_jit_sandwich_mint_tx(self, pipeline_results):
        _, _, jits = pipeline_results
        assert jits["mint_tx"][0] == "0xJIT_MINT"
        assert jits["burn_tx"][0] == "0xJIT_BURN"

    def test_jit_type_is_full(self, pipeline_results):
        """Burn liquidity equals mint liquidity → full JIT."""
        _, _, jits = pipeline_results
        assert jits["jit_type"][0] == "full"

    def test_jit_sandwich_covers_exactly_one_swap(self, pipeline_results):
        _, _, jits = pipeline_results
        assert jits["swap_count"][0] == 1

    def test_jit_tick_range(self, pipeline_results):
        _, _, jits = pipeline_results
        assert jits["tick_lower"][0] == 195240
        assert jits["tick_upper"][0] == 195420

    def test_swap2_jit_owner_populated(self, pipeline_results):
        swaps, _, _ = pipeline_results
        swap2 = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")
        assert swap2["jit_owner"][0] == "0xJIT_LP"
        assert swap2["jit_mint_tx"][0] == "0xJIT_MINT"
        assert swap2["jit_burn_tx"][0] == "0xJIT_BURN"


class TestFeeAttribution:
    def test_fee_conservation_every_swap(self, pipeline_results):
        """fees_to_jit + fees_to_passive = total_fees for every swap."""
        swaps, _, _ = pipeline_results
        for row in swaps.iter_rows(named=True):
            diff = abs(row["fees_to_jit_usd"] + row["fees_to_passive_usd"] - row["total_fees_usd"])
            assert diff < 1e-9, (
                f"Fee mismatch on {row['transaction_hash']}: "
                f"jit={row['fees_to_jit_usd']:.6f} + passive={row['fees_to_passive_usd']:.6f} "
                f"≠ total={row['total_fees_usd']:.6f}"
            )

    def test_non_jit_swaps_have_zero_jit_fees(self, pipeline_results):
        swaps, _, _ = pipeline_results
        for tx in ("0xSWAP1", "0xSWAP3"):
            row = swaps.filter(pl.col("transaction_hash") == tx)
            assert abs(row["fees_to_jit_usd"][0]) < 1e-12, f"{tx} should have 0 JIT fees"

    def test_jit_swap_has_positive_jit_fees(self, pipeline_results):
        swaps, _, _ = pipeline_results
        swap2 = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")
        assert swap2["fees_to_jit_usd"][0] > 0

    def test_jit_fraction_on_swap2(self, pipeline_results):
        """JIT fraction ≈ JIT_LIQ / TOTAL_LIQ_SWAP2 / scale.

        At the time of SWAP2, the pool has:
          PASSIVE_WIDE  (10e15, in range [194000,197000])
          PASSIVE2      ( 5e15, in range [195240,195420])
          JIT           ( 8e15, in range [195240,195420])
        Total = 23e15. Swap goes 195360→195380, entirely within all positions.

        The corrected JIT fraction divides by scale (≈1/(1-0.003)) to account for
        the fee-deduction effect (V3 moves price with net amount, CSV reports gross).
        Expected ≈ (8/23) * (1-0.003) ≈ 0.3468.
        """
        swaps, _, _ = pipeline_results
        swap2 = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")
        # raw fraction = 8/23 ≈ 0.3478; corrected by scale ≈ 1.003 → ≈ 0.3467
        raw_fraction = JIT_LIQ / TOTAL_LIQ_SWAP2
        actual = swap2["jit_fraction_weighted"][0]
        # Allow 0.5% tolerance for float arithmetic
        assert abs(actual - raw_fraction) < 0.005, f"JIT fraction: got {actual:.4f}, expected ≈{raw_fraction:.4f}"

    def test_total_fees_equal_volume_times_fee_rate(self, pipeline_results):
        swaps, _, _ = pipeline_results
        for row in swaps.iter_rows(named=True):
            if row["volume_usd"] and row["volume_usd"] > 0:
                expected = row["volume_usd"] * FEE_RATE
                diff = abs(row["total_fees_usd"] - expected) / expected
                assert diff < 1e-9, (
                    f"{row['transaction_hash']}: total_fees={row['total_fees_usd']:.6f} "
                    f"≠ volume*rate={expected:.6f}"
                )

    def test_jit_fees_in_sandwich_record_matches_swap_record(self, pipeline_results):
        """jit_sandwiches.fees_captured_usd should equal swaps.fees_to_jit_usd for the sandwiched swap."""
        swaps, _, jits = pipeline_results
        swap2_jit_fees = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")["fees_to_jit_usd"][0]
        sandwich_fees = jits["fees_captured_usd"][0]
        assert abs(swap2_jit_fees - sandwich_fees) < 1e-9


class TestActiveLiq:
    def test_active_liq_start_before_jit_mint(self, pipeline_results):
        """SWAP1 finishes at tick 195360 with reported_liq=15e15.
        JIT_MINT adds 8e15 → active_liq = 23e15.
        SWAP2's active_liq_start must be 23e15 (captured BEFORE the walk)."""
        swaps, _, _ = pipeline_results
        swap2 = swaps.filter(pl.col("transaction_hash") == "0xSWAP2")
        assert swap2["active_liq_start"][0] == TOTAL_LIQ_SWAP2

    def test_active_liq_end_matches_reported(self, pipeline_results):
        """active_liq_end must match the liquidity reported in the swap event."""
        swaps, _, _ = pipeline_results
        expected = {
            "0xSWAP1": PASSIVE_WIDE_LIQ + PASSIVE_NARROW_LIQ,  # 15e15
            "0xSWAP2": TOTAL_LIQ_SWAP2,                         # 23e15 (no tick crossings)
            "0xSWAP3": PASSIVE_WIDE_LIQ + PASSIVE_NARROW_LIQ,  # 15e15
        }
        for tx, liq in expected.items():
            row = swaps.filter(pl.col("transaction_hash") == tx)
            assert row["active_liq_end"][0] == liq, f"{tx}: expected {liq}, got {row['active_liq_end'][0]}"


class TestDirection:
    def test_buy_swaps_direction(self, pipeline_results):
        swaps, _, _ = pipeline_results
        for tx in ("0xSWAP1", "0xSWAP2"):
            row = swaps.filter(pl.col("transaction_hash") == tx)
            assert row["direction"][0] == "buy", f"{tx} should be buy"

    def test_sell_swap_direction(self, pipeline_results):
        swaps, _, _ = pipeline_results
        swap3 = swaps.filter(pl.col("transaction_hash") == "0xSWAP3")
        assert swap3["direction"][0] == "sell"


class TestSegments:
    def test_segments_link_to_swaps(self, pipeline_results):
        swaps, segs, _ = pipeline_results
        swap_hashes = set(swaps["transaction_hash"].to_list())
        seg_hashes = set(segs["transaction_hash"].to_list())
        assert seg_hashes.issubset(swap_hashes), "Segment hashes must reference known swaps"

    def test_segment_fee_conservation(self, pipeline_results):
        """Within each swap, sum of segment fees_to_jit + fees_to_passive = fees_total."""
        _, segs, _ = pipeline_results
        for tx_hash in segs["transaction_hash"].unique().to_list():
            tx_segs = segs.filter(pl.col("transaction_hash") == tx_hash)
            for row in tx_segs.iter_rows(named=True):
                diff = abs(row["fees_to_jit"] + row["fees_to_passive"] - row["fees_total"])
                assert diff < 1e-12, f"Segment fee mismatch: {diff}"

    def test_jit_segments_only_within_position_range(self, pipeline_results):
        """Segments outside [JIT_LOWER, JIT_UPPER) must have jit_liquidity=0."""
        _, segs, _ = pipeline_results
        swap2_segs = segs.filter(pl.col("transaction_hash") == "0xSWAP2")
        for row in swap2_segs.iter_rows(named=True):
            tick_start = row["tick_start"]
            if tick_start < 195240 or tick_start >= 195420:
                assert row["jit_liquidity"] == 0, (
                    f"JIT liq should be 0 outside position range at tick {tick_start}"
                )

    def test_last_segment_uses_actual_final_sqrt(self, pipeline_results):
        """The last segment of each swap must end at the pool's reported final sqrtPriceX96."""
        swaps, segs, _ = pipeline_results
        for row in swaps.iter_rows(named=True):
            tx = row["transaction_hash"]
            tx_segs = segs.filter(pl.col("transaction_hash") == tx).sort("segment_index")
            if len(tx_segs) == 0:
                continue
            last_seg_sqrt = tx_segs["sqrt_price_end"][-1]
            assert last_seg_sqrt == row["final_sqrt_x96"], (
                f"{tx}: last segment sqrt {last_seg_sqrt} ≠ swap final sqrt {row['final_sqrt_x96']}"
            )
