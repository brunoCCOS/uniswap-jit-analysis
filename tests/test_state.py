"""Tests for src/state.py — PoolState tick delta and liquidity tracking."""

import pytest
from src.state import PoolState


def make_state(tick: int = 100) -> PoolState:
    s = PoolState()
    s.tick = tick
    s.sqrt_x96 = 1
    return s


# ── apply_mint ────────────────────────────────────────────────────────────────

class TestApplyMint:
    def test_in_range_increases_active_liq(self):
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        assert s.active_liq == 1000

    def test_out_of_range_below_does_not_change_active_liq(self):
        s = make_state(tick=50)
        s.apply_mint(100, 200, 1000)
        assert s.active_liq == 0

    def test_out_of_range_above_does_not_change_active_liq(self):
        s = make_state(tick=250)
        s.apply_mint(100, 200, 1000)
        assert s.active_liq == 0

    def test_tick_at_lower_boundary_is_in_range(self):
        """V3: position is active when tickLower <= currentTick < tickUpper."""
        s = make_state(tick=100)
        s.apply_mint(100, 200, 500)
        assert s.active_liq == 500

    def test_tick_at_upper_boundary_is_out_of_range(self):
        s = make_state(tick=200)
        s.apply_mint(100, 200, 500)
        assert s.active_liq == 0

    def test_tick_delta_set_correctly(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 999)
        assert s.tick_deltas[100] == 999
        assert s.tick_deltas[200] == -999

    def test_multiple_overlapping_positions_accumulate(self):
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        s.apply_mint(120, 180, 500)
        assert s.active_liq == 1500

    def test_non_overlapping_in_range_positions_accumulate(self):
        s = make_state(tick=150)
        s.apply_mint(100, 200, 300)
        s.apply_mint(130, 170, 700)
        assert s.active_liq == 1000

    def test_sorted_ticks_populated(self):
        s = make_state(tick=0)
        s.apply_mint(10, 20, 100)
        s.apply_mint(30, 40, 200)
        assert list(s._sorted_ticks) == [10, 20, 30, 40]


# ── apply_burn ────────────────────────────────────────────────────────────────

class TestApplyBurn:
    def test_in_range_burn_decreases_active_liq(self):
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        s.apply_burn(100, 200, 1000)
        assert s.active_liq == 0

    def test_partial_burn_reduces_correctly(self):
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        s.apply_burn(100, 200, 400)
        assert s.active_liq == 600

    def test_burn_removes_tick_deltas(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 500)
        s.apply_burn(100, 200, 500)
        assert 100 not in s.tick_deltas
        assert 200 not in s.tick_deltas

    def test_burn_removes_ticks_from_sorted_list(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 500)
        s.apply_burn(100, 200, 500)
        assert 100 not in s._sorted_ticks
        assert 200 not in s._sorted_ticks

    def test_out_of_range_burn_does_not_affect_active_liq(self):
        s = make_state(tick=50)
        s.apply_mint(100, 200, 1000)
        s.apply_burn(100, 200, 1000)
        assert s.active_liq == 0

    def test_partial_burn_net_delta_preserved(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 1000)
        s.apply_burn(100, 200, 600)
        assert s.tick_deltas[100] == 400
        assert s.tick_deltas[200] == -400


# ── cross_tick_up / cross_tick_down ───────────────────────────────────────────

class TestCrossTickUpDown:
    def test_cross_tick_up_entering_position(self):
        """Crossing tickLower going up: entering range, active_liq increases."""
        s = make_state(tick=50)
        s.apply_mint(100, 200, 1000)
        s.active_liq = 0  # simulate being below the range
        s.cross_tick_up(100)
        assert s.active_liq == 1000

    def test_cross_tick_up_exiting_position(self):
        """Crossing tickUpper going up: exiting range, active_liq decreases."""
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        s.active_liq = 1000
        s.cross_tick_up(200)
        assert s.active_liq == 0

    def test_cross_tick_down_exiting_position(self):
        """Crossing tickLower going down: exiting range, active_liq decreases."""
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        s.active_liq = 1000
        s.cross_tick_down(100)
        assert s.active_liq == 0

    def test_cross_tick_down_entering_position(self):
        """Crossing tickUpper going down: entering range, active_liq increases."""
        s = make_state(tick=250)
        s.apply_mint(100, 200, 1000)
        s.active_liq = 0
        s.cross_tick_down(200)
        assert s.active_liq == 1000

    def test_crossing_noninitialized_tick_is_noop(self):
        s = make_state(tick=0)
        s.active_liq = 500
        s.cross_tick_up(999)   # no position at 999
        assert s.active_liq == 500

    def test_multiple_positions_at_same_tick(self):
        """Two positions sharing tickLower: crossing once adds both."""
        s = make_state(tick=50)
        s.apply_mint(100, 300, 1000)
        s.apply_mint(100, 200, 500)
        s.active_liq = 0
        s.cross_tick_up(100)
        assert s.active_liq == 1500

    def test_round_trip_up_down(self):
        """Cross up then down returns active_liq to original."""
        s = make_state(tick=150)
        s.apply_mint(100, 200, 1000)
        original = s.active_liq  # 1000 since tick=150 is in range
        s.cross_tick_up(200)
        assert s.active_liq == 0
        s.cross_tick_down(200)
        assert s.active_liq == original


# ── ticks_between ─────────────────────────────────────────────────────────────

class TestTicksBetween:
    def _state_with_ticks(self, tick_liq_pairs: list[tuple[int, int]]) -> PoolState:
        s = make_state(tick=0)
        for (tl, tu), liq in tick_liq_pairs:
            s.apply_mint(tl, tu, liq)
        return s

    def test_upward_returns_ascending_ticks(self):
        s = self._state_with_ticks([((10, 50), 100), ((20, 40), 200)])
        result = s.ticks_between(0, 60)
        assert result == sorted(result)

    def test_downward_returns_descending_ticks(self):
        s = self._state_with_ticks([((10, 50), 100), ((20, 40), 200)])
        result = s.ticks_between(60, 0)
        assert result == sorted(result, reverse=True)

    def test_tick_end_always_appended(self):
        s = self._state_with_ticks([((10, 50), 100)])
        result = s.ticks_between(0, 100)
        assert result[-1] == 100

    def test_tick_start_excluded(self):
        s = self._state_with_ticks([((10, 50), 100)])
        result = s.ticks_between(10, 50)
        assert 10 not in result

    def test_empty_range_returns_only_end(self):
        s = make_state(tick=0)
        result = s.ticks_between(0, 100)
        assert result == [100]

    def test_downward_end_tick_is_first_returned(self):
        s = self._state_with_ticks([((10, 50), 100), ((20, 40), 200)])
        result = s.ticks_between(60, 5)
        assert result[-1] == 5

    def test_upward_only_initialized_ticks_included(self):
        s = self._state_with_ticks([((10, 50), 100)])
        result = s.ticks_between(0, 100)
        # 10 and 50 are initialized; also appends 100
        assert set(result) == {10, 50, 100}

    def test_downward_only_initialized_ticks_included(self):
        s = self._state_with_ticks([((10, 50), 100)])
        result = s.ticks_between(100, 0)
        assert set(result) == {50, 10, 0}


# ── apply_swap ────────────────────────────────────────────────────────────────

class TestApplySwap:
    def test_updates_tick_and_sqrt(self):
        s = make_state(tick=100)
        s.apply_swap(200, 999, 500)
        assert s.tick == 200
        assert s.sqrt_x96 == 999

    def test_uses_reported_liq_as_ground_truth(self):
        s = make_state(tick=100)
        s.active_liq = 999  # stale value
        s.apply_swap(200, 1, reported_liq=12345)
        assert s.active_liq == 12345

    def test_apply_swap_corrects_drift(self):
        """After a segment walk that may have drifted, apply_swap corrects active_liq."""
        s = make_state(tick=0)
        s.apply_mint(0, 100, 1000)
        s.apply_mint(50, 150, 500)
        # Manually drift active_liq
        s.active_liq = 99999
        s.apply_swap(120, 1, reported_liq=500)
        assert s.active_liq == 500


# ── _add_delta (internal): tick cleanup ───────────────────────────────────────

class TestAddDelta:
    def test_net_zero_delta_removes_tick(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 500)
        assert 100 in s.tick_deltas
        s.apply_burn(100, 200, 500)
        assert 100 not in s.tick_deltas
        assert 100 not in s._sorted_ticks

    def test_net_zero_delta_on_both_bounds(self):
        s = make_state(tick=0)
        s.apply_mint(100, 200, 300)
        s.apply_burn(100, 200, 300)
        assert len(s.tick_deltas) == 0
        assert len(s._sorted_ticks) == 0
