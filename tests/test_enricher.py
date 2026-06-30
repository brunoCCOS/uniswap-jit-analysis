"""Tests for src/enricher.py — swap handling, fee attribution, active_liq_start timing."""

import pytest
from src.config import PoolConfig
from src.enricher import _handle_swap, _walk_segments, enrich_pool
from src.price import Q96, tick_to_sqrt_x96
from src.state import PoolState

import polars as pl
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _usdc_weth_cfg() -> PoolConfig:
    return PoolConfig(
        pool_id="test",
        pool_dir=Path("/tmp"),
        token0_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        token1_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        tick_spacing=60,
        fee_millionths=3000,
    )


def _state_at(tick: int, liq: int = 0) -> PoolState:
    s = PoolState()
    s.tick = tick
    s.sqrt_x96 = tick_to_sqrt_x96(tick)
    s.active_liq = liq
    return s


def _swap_row(
    block: int = 1000,
    tx_index: int = 1,
    tx_hash: str = "0xSWAP",
    tick: int = 200000,
    sqrt_x96: int | None = None,
    liq: int = 1_000_000,
    a0: float = -1_000_000_000.0,  # USDC out (negative)
    a1: float = 1_000_000_000_000_000_000.0,  # WETH in (positive)
    p0: float = 1.0,
    p1: float = 2000.0,
) -> dict:
    return {
        "event": "swap",
        "block_number": block,
        "transaction_index": tx_index,
        "log_index": 1,
        "transaction_hash": tx_hash,
        "sender_address": "0xSENDER",
        "recipient_address": "0xRECIP",
        "txFrom": "0xFROM",
        "amount0": a0,
        "amount1": a1,
        "token0_price_usd": p0,
        "token1_price_usd": p1,
        "liquidity": float(liq),
        "sqrtPriceX96": str(sqrt_x96 or tick_to_sqrt_x96(tick)),
        "tick": float(tick),
        "tickLower": None,
        "tickUpper": None,
        "timestamp": "2021-01-01",
    }


# ── active_liq_start captured before walk ────────────────────────────────────

class TestActiveLiqStart:
    def test_active_liq_start_is_pre_walk_value(self):
        """active_liq_start must reflect liquidity BEFORE any tick crossings."""
        cfg = _usdc_weth_cfg()
        # Set up two positions so ticks are crossed during the walk
        state = _state_at(tick=195300, liq=0)
        state.apply_mint(195240, 195360, 500_000)  # in range
        state.apply_mint(195300, 195420, 750_000)  # in range
        # active_liq = 1_250_000

        initial_liq = state.active_liq
        assert initial_liq == 1_250_000

        swap_records: list[dict] = []
        segment_records: list[dict] = []
        jit_map: dict = {}

        row = _swap_row(
            tick=195420,
            sqrt_x96=tick_to_sqrt_x96(195420),
            liq=750_000,   # after crossing 195360, only the second position remains
        )
        _handle_swap(row, state, cfg, jit_map, swap_records, segment_records)

        rec = swap_records[0]
        assert rec["active_liq_start"] == initial_liq, (
            f"active_liq_start={rec['active_liq_start']}, expected {initial_liq}"
        )

    def test_active_liq_end_is_reported_liq(self):
        """active_liq_end must equal the on-chain reported liquidity from the swap event."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        swap_records: list[dict] = []
        row = _swap_row(tick=195360, liq=500_000)
        _handle_swap(row, state, cfg, {}, swap_records, [])
        assert swap_records[0]["active_liq_end"] == 500_000


# ── Direction detection ───────────────────────────────────────────────────────

class TestDirection:
    def test_buy_when_sqrt_increases(self):
        """When sqrtPrice goes up (more WETH per USDC), direction = 'buy'."""
        cfg = _usdc_weth_cfg()
        initial_tick = 195300
        state = _state_at(tick=initial_tick, liq=1_000_000)
        sqrt_final = tick_to_sqrt_x96(195360)  # higher tick = higher sqrtPrice
        row = _swap_row(tick=195300, sqrt_x96=sqrt_final, liq=1_000_000)
        # tick unchanged (intra-tick) but sqrt increased
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        assert swap_records[0]["direction"] == "buy"

    def test_sell_when_sqrt_decreases(self):
        """When sqrtPrice goes down, direction = 'sell'."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195360, liq=1_000_000)
        sqrt_final = tick_to_sqrt_x96(195300)  # lower sqrtPrice
        row = _swap_row(tick=195360, sqrt_x96=sqrt_final, liq=1_000_000)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        assert swap_records[0]["direction"] == "sell"

    def test_intra_tick_direction_from_sqrt_not_tick(self):
        """Intra-tick swap: tick unchanged, direction derived from sqrt price."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        # Both initial and final tick = 195300, but sqrtPrice moves up
        sqrt_up = tick_to_sqrt_x96(195300) + 10**20
        row = _swap_row(tick=195300, sqrt_x96=sqrt_up, liq=1_000_000)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        assert swap_records[0]["direction"] == "buy"


# ── Volume calculation ────────────────────────────────────────────────────────

class TestVolume:
    def test_volume_uses_positive_amount0(self):
        """If a0 > 0 (token0 = USDC in), volume = a0 * price0 / 10^decimals."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        # a0 = 1_000_000_000 USDC_smallest (= 1000 USDC at $1 each = $1000)
        # a1 = -500_000_000_000_000_000 WETH_smallest (output, negative)
        row = _swap_row(tick=195360, a0=1_000_000_000.0, a1=-500_000_000_000_000_000.0, p0=1.0, p1=2000.0)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        # volume = 1e9 / 10^6 * 1.0 = 1000.0 USD
        assert abs(swap_records[0]["volume_usd"] - 1000.0) < 1.0

    def test_volume_uses_positive_amount1(self):
        """If a0 < 0 and a1 > 0, volume = a1 * price1 / 10^decimals."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        # a0 = -999_000_000 USDC_smallest (output), a1 = 1_000_000_000_000_000 WETH_smallest
        # 1e15 WETH_smallest * $2000/WETH / 1e18 = $2.0
        row = _swap_row(tick=195300, sqrt_x96=tick_to_sqrt_x96(195240),
                        a0=-999_000_000.0, a1=1_000_000_000_000_000.0, p0=1.0, p1=2000.0)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        # volume = 1e15 / 1e18 * 2000.0 = 2.0 USD
        assert abs(swap_records[0]["volume_usd"] - 2.0) < 0.1

    def test_volume_zero_when_both_amounts_nonpositive(self):
        """Degenerate case: a0 <= 0 and a1 <= 0 → volume = 0."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        row = _swap_row(tick=195300, a0=-100.0, a1=-100.0)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        assert swap_records[0]["volume_usd"] == 0.0


# ── Fee attribution ───────────────────────────────────────────────────────────

class TestFees:
    def test_total_fees_equal_volume_times_fee_rate(self):
        """total_fees_usd = volume_usd * fee_rate (0.3% for this pool)."""
        cfg = _usdc_weth_cfg()  # fee_millionths=3000 → 0.3%
        state = _state_at(tick=195300, liq=1_000_000)
        row = _swap_row(tick=195360, a0=1_000_000_000.0, a1=-5e17, p0=1.0, p1=2000.0)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        rec = swap_records[0]
        expected_fees = rec["volume_usd"] * 0.003
        assert abs(rec["total_fees_usd"] - expected_fees) < 1e-10

    def test_fees_to_jit_plus_passive_equals_total(self):
        """fees_to_jit + fees_to_passive = total_fees (no leakage)."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        row = _swap_row(tick=195360, a0=1_000_000_000.0, a1=-5e17)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        rec = swap_records[0]
        assert abs(rec["fees_to_jit_usd"] + rec["fees_to_passive_usd"] - rec["total_fees_usd"]) < 1e-9

    def test_no_jit_all_fees_to_passive(self):
        """Without a JIT sandwich, all fees go to passive LPs."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)
        row = _swap_row(tick=195360, a0=1_000_000_000.0, a1=-5e17)
        swap_records: list[dict] = []
        _handle_swap(row, state, cfg, {}, swap_records, [])
        rec = swap_records[0]
        assert rec["fees_to_jit_usd"] == 0.0 or abs(rec["fees_to_jit_usd"]) < 1e-12
        assert abs(rec["fees_to_passive_usd"] - rec["total_fees_usd"]) < 1e-9


# ── Walk segments ─────────────────────────────────────────────────────────────

class TestWalkSegments:
    def test_no_segments_when_ticks_equal(self):
        state = _state_at(tick=100, liq=1_000_000)
        result = _walk_segments(
            state=state,
            initial_sqrt=tick_to_sqrt_x96(100),
            final_sqrt=tick_to_sqrt_x96(100),
            initial_tick=100,
            final_tick=100,
            direction_up=True,
            jit=None,
            cfg=_usdc_weth_cfg(),
            fee_rate=0.003,
            p0=1.0,
            p1=2000.0,
            tx_hash="0xTEST",
            segment_records=[],
        )
        assert result == []

    def test_segment_count_matches_tick_crossings(self):
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=500_000)
        # Add positions at 195360 and 195420 so those ticks are initialized
        state.apply_mint(195360, 195480, 100_000)
        state.apply_mint(195420, 195540, 100_000)

        initial_sqrt = tick_to_sqrt_x96(195300)
        final_sqrt = tick_to_sqrt_x96(195480)
        seg_recs: list[dict] = []
        segs = _walk_segments(
            state=state,
            initial_sqrt=initial_sqrt,
            final_sqrt=final_sqrt,
            initial_tick=195300,
            final_tick=195480,
            direction_up=True,
            jit=None,
            cfg=cfg,
            fee_rate=0.003,
            p0=1.0,
            p1=2000.0,
            tx_hash="0xTEST",
            segment_records=seg_recs,
        )
        # Segments: [195300→195360], [195360→195420], [195420→195480]
        assert len(segs) == 3
        assert len(seg_recs) == 3

    def test_last_segment_uses_actual_final_sqrt(self):
        """Last segment boundary must use the actual on-chain final_sqrt, not tick formula."""
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=500_000)
        state.apply_mint(195360, 195480, 100_000)

        final_sqrt = tick_to_sqrt_x96(195480) + 12345  # slightly off from exact tick
        segs = _walk_segments(
            state=state,
            initial_sqrt=tick_to_sqrt_x96(195300),
            final_sqrt=final_sqrt,
            initial_tick=195300,
            final_tick=195480,
            direction_up=True,
            jit=None,
            cfg=cfg,
            fee_rate=0.003,
            p0=1.0,
            p1=2000.0,
            tx_hash="0xTEST",
            segment_records=[],
        )
        assert int(segs[-1]["sqrt_price_end"]) == final_sqrt

    def test_jit_fees_proportional_to_liquidity(self):
        """JIT liquidity = 50% of total → JIT captures ~50% of fees."""
        from src.detector import JITSandwich
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)

        jit = JITSandwich(
            block_number=1000,
            owner="0xJIT",
            mint_tx="0xMINT",
            mint_tx_index=1,
            burn_tx="0xBURN",
            burn_tx_index=3,
            tick_lower=195240,
            tick_upper=195420,
            jit_liquidity=500_000,   # 50% of 1_000_000 total
            burn_liquidity=500_000,
            jit_type="full",
            new_passive_liq=0,
            swap_tx_hashes=["0xSWAP"],
        )

        segs = _walk_segments(
            state=state,
            initial_sqrt=tick_to_sqrt_x96(195300),
            final_sqrt=tick_to_sqrt_x96(195360),
            initial_tick=195300,
            final_tick=195360,
            direction_up=True,
            jit=jit,
            cfg=cfg,
            fee_rate=0.003,
            p0=1.0,
            p1=2000.0,
            tx_hash="0xSWAP",
            segment_records=[],
        )
        assert segs
        for seg in segs:
            if seg["total_liquidity"] > 0:
                ratio = seg["fees_to_jit"] / seg["fees_total"]
                assert abs(ratio - 0.5) < 0.01, f"Expected ~50% JIT fee share, got {ratio:.3f}"

    def test_jit_out_of_range_gets_no_fees(self):
        """JIT position outside the current tick range gets zero fees."""
        from src.detector import JITSandwich
        cfg = _usdc_weth_cfg()
        state = _state_at(tick=195300, liq=1_000_000)

        jit = JITSandwich(
            block_number=1000,
            owner="0xJIT",
            mint_tx="0xMINT",
            mint_tx_index=1,
            burn_tx="0xBURN",
            burn_tx_index=3,
            tick_lower=196000,   # well above the swap range
            tick_upper=197000,
            jit_liquidity=500_000,
            burn_liquidity=500_000,
            jit_type="full",
            new_passive_liq=0,
            swap_tx_hashes=["0xSWAP"],
        )
        segs = _walk_segments(
            state=state,
            initial_sqrt=tick_to_sqrt_x96(195300),
            final_sqrt=tick_to_sqrt_x96(195360),
            initial_tick=195300,
            final_tick=195360,
            direction_up=True,
            jit=jit,
            cfg=cfg,
            fee_rate=0.003,
            p0=1.0,
            p1=2000.0,
            tx_hash="0xSWAP",
            segment_records=[],
        )
        for seg in segs:
            assert seg["fees_to_jit"] == 0.0
            assert seg["jit_liquidity"] == 0


# ── enrich_pool end-to-end ────────────────────────────────────────────────────

class TestEnrichPool:
    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows, infer_schema_length=None)

    def _base_rows(self) -> list[dict]:
        """Minimal valid event sequence: initialize → mint → swap → burn."""
        sqrt_init = tick_to_sqrt_x96(195300)
        return [
            {
                "block_number": 1000, "timestamp": "2021-01-01", "log_index": 0,
                "transaction_index": 0, "transaction_hash": "0xINIT",
                "event": "initialize", "owner_address": None,
                "sender_address": None, "recipient_address": None, "txFrom": None,
                "amount0": None, "token0_price_usd": None,
                "amount1": None, "token1_price_usd": None,
                "liquidity": None, "sqrtPriceX96": str(sqrt_init),
                "tick": 195300.0, "tickLower": None, "tickUpper": None,
            },
            {
                "block_number": 1000, "timestamp": "2021-01-01", "log_index": 1,
                "transaction_index": 1, "transaction_hash": "0xMINT",
                "event": "mint", "owner_address": "0xLP",
                "sender_address": None, "recipient_address": None, "txFrom": None,
                "amount0": None, "token0_price_usd": None,
                "amount1": None, "token1_price_usd": None,
                "liquidity": 2_000_000.0, "sqrtPriceX96": None,
                "tick": None, "tickLower": 195240.0, "tickUpper": 195420.0,
            },
            {
                "block_number": 1001, "timestamp": "2021-01-02", "log_index": 0,
                "transaction_index": 0, "transaction_hash": "0xSWAP",
                "event": "swap", "owner_address": None,
                "sender_address": "0xSENDER", "recipient_address": "0xRECIP", "txFrom": "0xFROM",
                "amount0": 1_000_000_000.0, "token0_price_usd": 1.0,
                "amount1": -5e17, "token1_price_usd": 2000.0,
                "liquidity": 2_000_000.0, "sqrtPriceX96": str(tick_to_sqrt_x96(195360)),
                "tick": 195360.0, "tickLower": None, "tickUpper": None,
            },
        ]

    def test_returns_three_dataframes(self):
        df = self._make_df(self._base_rows())
        swaps, segs, jits = enrich_pool(df, _usdc_weth_cfg())
        assert isinstance(swaps, pl.DataFrame)
        assert isinstance(segs, pl.DataFrame)
        assert isinstance(jits, pl.DataFrame)

    def test_one_swap_produces_one_swap_record(self):
        df = self._make_df(self._base_rows())
        swaps, _, _ = enrich_pool(df, _usdc_weth_cfg())
        assert len(swaps) == 1

    def test_swap_record_has_required_columns(self):
        df = self._make_df(self._base_rows())
        swaps, _, _ = enrich_pool(df, _usdc_weth_cfg())
        required = [
            "transaction_hash", "direction", "volume_usd",
            "total_fees_usd", "fees_to_jit_usd", "fees_to_passive_usd",
            "active_liq_start", "active_liq_end", "is_jit",
        ]
        for col in required:
            assert col in swaps.columns, f"Missing column: {col}"

    def test_no_jit_in_simple_swap(self):
        df = self._make_df(self._base_rows())
        swaps, _, jits = enrich_pool(df, _usdc_weth_cfg())
        assert swaps["is_jit"][0] == False
        assert len(jits) == 0

    def test_jit_sandwich_detected_and_recorded(self):
        rows = self._base_rows()
        sqrt_init = tick_to_sqrt_x96(195300)
        sqrt_final = tick_to_sqrt_x96(195360)
        # Build a full JIT sandwich: same block as swap for detection to fire
        jit_block_rows = [
            {
                "block_number": 2000, "timestamp": "2021-01-03", "log_index": 0,
                "transaction_index": 0, "transaction_hash": "0xJMINT",
                "event": "mint", "owner_address": "0xJIT",
                "sender_address": None, "recipient_address": None, "txFrom": None,
                "amount0": None, "token0_price_usd": None,
                "amount1": None, "token1_price_usd": None,
                "liquidity": 500_000.0, "sqrtPriceX96": None,
                "tick": None, "tickLower": 195240.0, "tickUpper": 195420.0,
            },
            {
                "block_number": 2000, "timestamp": "2021-01-03", "log_index": 1,
                "transaction_index": 1, "transaction_hash": "0xJSWAP",
                "event": "swap", "owner_address": None,
                "sender_address": "0xSENDER", "recipient_address": "0xRECIP", "txFrom": "0xFROM",
                "amount0": 500_000_000.0, "token0_price_usd": 1.0,
                "amount1": -2e17, "token1_price_usd": 2000.0,
                "liquidity": 2_500_000.0, "sqrtPriceX96": str(sqrt_final),
                "tick": 195360.0, "tickLower": None, "tickUpper": None,
            },
            {
                "block_number": 2000, "timestamp": "2021-01-03", "log_index": 2,
                "transaction_index": 2, "transaction_hash": "0xJBURN",
                "event": "burn", "owner_address": "0xJIT",
                "sender_address": None, "recipient_address": None, "txFrom": None,
                "amount0": None, "token0_price_usd": None,
                "amount1": None, "token1_price_usd": None,
                "liquidity": 500_000.0, "sqrtPriceX96": None,
                "tick": None, "tickLower": 195240.0, "tickUpper": 195420.0,
            },
        ]
        df = self._make_df(rows + jit_block_rows)
        swaps, _, jits = enrich_pool(df, _usdc_weth_cfg())
        jit_swaps = swaps.filter(pl.col("is_jit"))
        assert len(jit_swaps) == 1
        assert len(jits) == 1
        assert jits["owner"][0] == "0xJIT"

    def test_fees_conservation_across_all_swaps(self):
        """Sum of fees_to_jit + fees_to_passive = sum of total_fees across all swaps."""
        df = self._make_df(self._base_rows())
        swaps, _, _ = enrich_pool(df, _usdc_weth_cfg())
        total = swaps["total_fees_usd"].sum()
        split = swaps["fees_to_jit_usd"].sum() + swaps["fees_to_passive_usd"].sum()
        assert abs(total - split) < 1e-9

    def test_empty_dataframe_returns_empty_results(self):
        df = pl.DataFrame(schema={
            "block_number": pl.Int64, "timestamp": pl.Utf8, "log_index": pl.Int32,
            "transaction_index": pl.Int32, "transaction_hash": pl.Utf8,
            "event": pl.Utf8, "owner_address": pl.Utf8, "sender_address": pl.Utf8,
            "recipient_address": pl.Utf8, "txFrom": pl.Utf8,
            "amount0": pl.Float64, "token0_price_usd": pl.Float64,
            "amount1": pl.Float64, "token1_price_usd": pl.Float64,
            "liquidity": pl.Float64, "sqrtPriceX96": pl.Utf8,
            "tick": pl.Float64, "tickLower": pl.Float64, "tickUpper": pl.Float64,
        })
        swaps, segs, jits = enrich_pool(df, _usdc_weth_cfg())
        assert len(swaps) == 0
        assert len(segs) == 0
        assert len(jits) == 0
