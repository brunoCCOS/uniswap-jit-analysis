"""Tests for src/detector.py — JIT sandwich detection."""

import pytest
from src.detector import JITSandwich, detect_jit


def _row(
    event: str,
    tx_index: int,
    block: int = 1000,
    owner: str = "0xABC",
    tick_lower: float = 100.0,
    tick_upper: float = 200.0,
    liquidity: float = 1_000_000.0,
    tx_hash: str | None = None,
    sender: str = "0xSENDER",
    recipient: str = "0xRECIPIENT",
) -> dict:
    return {
        "event": event,
        "block_number": block,
        "transaction_index": tx_index,
        "log_index": 0,
        "transaction_hash": tx_hash or f"0x{event[:3].upper()}{tx_index:04d}",
        "owner_address": owner if event in ("mint", "burn") else None,
        "sender_address": sender if event == "swap" else None,
        "recipient_address": recipient if event == "swap" else None,
        "tickLower": tick_lower if event in ("mint", "burn") else None,
        "tickUpper": tick_upper if event in ("mint", "burn") else None,
        "liquidity": liquidity,
        "sqrtPriceX96": "79228162514264337593543950336",
        "tick": 150.0,
    }


# ── Basic detection ───────────────────────────────────────────────────────────

class TestDetectJITBasic:
    def test_empty_block_returns_empty(self):
        assert detect_jit([]) == {}

    def test_no_swaps_returns_empty(self):
        rows = [_row("mint", 1), _row("burn", 3)]
        assert detect_jit(rows) == {}

    def test_no_mints_returns_empty(self):
        rows = [_row("swap", 1, tx_hash="0xSWAP"), _row("burn", 2)]
        assert detect_jit(rows) == {}

    def test_no_burns_returns_empty(self):
        rows = [_row("mint", 1), _row("swap", 2, tx_hash="0xSWAP")]
        assert detect_jit(rows) == {}

    def test_basic_sandwich_detected(self):
        rows = [
            _row("mint", 1, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, tx_hash="0xBURN"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP" in result
        s = result["0xSWAP"]
        assert s.mint_tx == "0xMINT"
        assert s.burn_tx == "0xBURN"
        assert s.jit_liquidity == 1_000_000

    def test_swap_before_mint_not_sandwiched(self):
        rows = [
            _row("swap", 1, tx_hash="0xSWAP"),
            _row("mint", 2, tx_hash="0xMINT"),
            _row("burn", 3, tx_hash="0xBURN"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP" not in result

    def test_swap_after_burn_not_sandwiched(self):
        rows = [
            _row("mint", 1, tx_hash="0xMINT"),
            _row("burn", 2, tx_hash="0xBURN"),
            _row("swap", 3, tx_hash="0xSWAP"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP" not in result


# ── Ownership matching ────────────────────────────────────────────────────────

class TestOwnerMatching:
    def test_different_owner_burn_not_matched(self):
        rows = [
            _row("mint", 1, owner="0xALICE", tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, owner="0xBOB", tx_hash="0xBURN"),
        ]
        assert detect_jit(rows) == {}

    def test_same_owner_required(self):
        rows = [
            _row("mint", 1, owner="0xALICE", tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, owner="0xALICE", tx_hash="0xBURN"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP" in result

    def test_empty_owner_skipped(self):
        rows = [
            _row("mint", 1, owner="", tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, owner="", tx_hash="0xBURN"),
        ]
        assert detect_jit(rows) == {}


# ── Tick range matching ───────────────────────────────────────────────────────

class TestTickRangeMatching:
    def test_different_tick_range_not_matched(self):
        rows = [
            _row("mint", 1, tick_lower=100.0, tick_upper=200.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, tick_lower=150.0, tick_upper=250.0, tx_hash="0xBURN"),
        ]
        assert detect_jit(rows) == {}

    def test_exact_tick_range_required(self):
        rows = [
            _row("mint", 1, tick_lower=100.0, tick_upper=200.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, tick_lower=100.0, tick_upper=200.0, tx_hash="0xBURN"),
        ]
        assert "0xSWAP" in detect_jit(rows)


# ── JIT type classification ───────────────────────────────────────────────────

class TestJITType:
    def test_full_jit_when_burn_equals_mint(self):
        rows = [
            _row("mint", 1, liquidity=1_000_000.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, liquidity=1_000_000.0, tx_hash="0xBURN"),
        ]
        s = detect_jit(rows)["0xSWAP"]
        assert s.jit_type == "full"
        assert s.new_passive_liq == 0

    def test_partial_jit_when_burn_less_than_mint(self):
        rows = [
            _row("mint", 1, liquidity=1_000_000.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, liquidity=600_000.0, tx_hash="0xBURN"),
        ]
        s = detect_jit(rows)["0xSWAP"]
        assert s.jit_type == "partial"
        assert s.new_passive_liq == 400_000

    def test_full_jit_when_burn_exceeds_mint(self):
        """Burn > mint still classified as full (burn_liq >= mint_liq)."""
        rows = [
            _row("mint", 1, liquidity=1_000_000.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, liquidity=1_200_000.0, tx_hash="0xBURN"),
        ]
        s = detect_jit(rows)["0xSWAP"]
        assert s.jit_type == "full"
        assert s.new_passive_liq == 0  # max(0, mint - burn) = 0


# ── Multi-swap sandwiches ─────────────────────────────────────────────────────

class TestMultiSwap:
    def test_multiple_swaps_all_sandwiched(self):
        rows = [
            _row("mint", 1, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP1"),
            _row("swap", 3, tx_hash="0xSWAP2"),
            _row("swap", 4, tx_hash="0xSWAP3"),
            _row("burn", 5, tx_hash="0xBURN"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP1" in result
        assert "0xSWAP2" in result
        assert "0xSWAP3" in result
        # All map to the same sandwich
        assert result["0xSWAP1"].mint_tx == result["0xSWAP2"].mint_tx

    def test_swap_count_in_sandwich(self):
        rows = [
            _row("mint", 1, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP1"),
            _row("swap", 3, tx_hash="0xSWAP2"),
            _row("burn", 4, tx_hash="0xBURN"),
        ]
        s = detect_jit(rows)["0xSWAP1"]
        assert len(s.swap_tx_hashes) == 2


# ── Greedy burn matching ──────────────────────────────────────────────────────

class TestGreedyMatching:
    def test_burn_only_used_once(self):
        """A burn matched to mint1 cannot be reused by mint2."""
        rows = [
            _row("mint", 1, owner="0xA", tx_hash="0xMINT1"),
            _row("swap", 2, tx_hash="0xSWAP1"),
            _row("mint", 3, owner="0xA", tx_hash="0xMINT2"),
            _row("swap", 4, tx_hash="0xSWAP2"),
            _row("burn", 5, owner="0xA", tx_hash="0xBURN1"),
        ]
        result = detect_jit(rows)
        # MINT1(tx=1) grabs BURN1(tx=5) first; both SWAP1 and SWAP2 fall in [1,5] window.
        # MINT2(tx=3) gets no burn → no distinct sandwich for it.
        assert "0xSWAP1" in result
        assert "0xSWAP2" in result  # covered by MINT1's sandwich
        # Both swaps attribute to the same sandwich (MINT1, not MINT2)
        assert result["0xSWAP1"].mint_tx == result["0xSWAP2"].mint_tx == "0xMINT1"

    def test_second_mint_gets_no_burn_when_burned_already_used(self):
        """When only one burn exists and is used by mint1, mint2 cannot form a sandwich."""
        rows = [
            _row("mint", 1, owner="0xA", tick_lower=100.0, tick_upper=200.0, tx_hash="0xMINT1"),
            _row("swap", 2, tx_hash="0xSWAP1"),
            _row("burn", 3, owner="0xA", tick_lower=100.0, tick_upper=200.0, tx_hash="0xBURN1"),
            _row("mint", 4, owner="0xA", tick_lower=100.0, tick_upper=200.0, tx_hash="0xMINT2"),
            _row("swap", 5, tx_hash="0xSWAP2"),
            # No second burn after MINT2
        ]
        result = detect_jit(rows)
        assert "0xSWAP1" in result                      # MINT1→BURN1 works
        assert "0xSWAP2" not in result                  # MINT2 has no burn, never sandwiched
        assert result["0xSWAP1"].mint_tx == "0xMINT1"

    def test_two_independent_sandwiches_different_owners(self):
        rows = [
            _row("mint", 1, owner="0xA", tx_hash="0xMINT1"),
            _row("swap", 2, tx_hash="0xSWAP1"),
            _row("burn", 3, owner="0xA", tx_hash="0xBURN1"),
            _row("mint", 4, owner="0xB", tx_hash="0xMINT2"),
            _row("swap", 5, tx_hash="0xSWAP2"),
            _row("burn", 6, owner="0xB", tx_hash="0xBURN2"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP1" in result
        assert "0xSWAP2" in result
        assert result["0xSWAP1"].owner == "0xA"
        assert result["0xSWAP2"].owner == "0xB"

    def test_two_sandwiches_different_tick_ranges(self):
        rows = [
            _row("mint", 1, owner="0xA", tick_lower=100.0, tick_upper=200.0, tx_hash="0xMINTA"),
            _row("mint", 2, owner="0xA", tick_lower=300.0, tick_upper=400.0, tx_hash="0xMINTB"),
            _row("swap", 3, tx_hash="0xSWAP"),
            _row("burn", 4, owner="0xA", tick_lower=100.0, tick_upper=200.0, tx_hash="0xBURNA"),
            _row("burn", 5, owner="0xA", tick_lower=300.0, tick_upper=400.0, tx_hash="0xBURNB"),
        ]
        result = detect_jit(rows)
        assert "0xSWAP" in result


# ── Result fields ─────────────────────────────────────────────────────────────

class TestResultFields:
    def test_sandwich_fields_populated(self):
        rows = [
            _row("mint", 1, block=5000, owner="0xJIT", tick_lower=50.0, tick_upper=150.0,
                 liquidity=2_000_000.0, tx_hash="0xMINT"),
            _row("swap", 2, tx_hash="0xSWAP"),
            _row("burn", 3, owner="0xJIT", tick_lower=50.0, tick_upper=150.0,
                 liquidity=2_000_000.0, tx_hash="0xBURN"),
        ]
        s = detect_jit(rows)["0xSWAP"]
        assert s.block_number == 5000
        assert s.owner == "0xJIT"
        assert s.tick_lower == 50
        assert s.tick_upper == 150
        assert s.jit_liquidity == 2_000_000
        assert s.burn_liquidity == 2_000_000
        assert s.mint_tx_index == 1
        assert s.burn_tx_index == 3
        assert "0xSWAP" in s.swap_tx_hashes
