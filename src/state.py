"""Pool state: tick-level liquidity map and running pool state."""

from collections import defaultdict
from dataclasses import dataclass, field

from sortedcontainers import SortedList


@dataclass
class PoolState:
    tick: int = 0
    sqrt_x96: int = 0           # current sqrtPriceX96 (Python big int)
    active_liq: int = 0         # current active liquidity

    # Sparse tick delta map: tick → net liquidity delta
    # +L at tickLower, -L at tickUpper per position (V3 convention)
    tick_deltas: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _sorted_ticks: SortedList = field(default_factory=SortedList)

    def initialized(self) -> bool:
        return self.sqrt_x96 != 0

    # ------------------------------------------------------------------
    # Tick delta maintenance
    # ------------------------------------------------------------------

    def apply_mint(self, tick_lower: int, tick_upper: int, liquidity: int) -> None:
        self._add_delta(tick_lower, liquidity)
        self._add_delta(tick_upper, -liquidity)
        if tick_lower <= self.tick < tick_upper:
            self.active_liq += liquidity

    def apply_burn(self, tick_lower: int, tick_upper: int, liquidity: int) -> None:
        self._add_delta(tick_lower, -liquidity)
        self._add_delta(tick_upper, liquidity)
        if tick_lower <= self.tick < tick_upper:
            self.active_liq -= liquidity

    def _add_delta(self, tick: int, delta: int) -> None:
        if tick not in self.tick_deltas:
            self._sorted_ticks.add(tick)
        self.tick_deltas[tick] += delta
        if self.tick_deltas[tick] == 0:
            del self.tick_deltas[tick]
            self._sorted_ticks.remove(tick)

    # ------------------------------------------------------------------
    # Tick crossing helpers for swap walks
    # ------------------------------------------------------------------

    def ticks_between(self, tick_start: int, tick_end: int) -> list[int]:
        """
        Return initialized ticks strictly between tick_start and tick_end
        (exclusive on start, inclusive on end), ordered in traversal direction.
        The final boundary tick_end is appended so callers can treat it as
        the last segment boundary even if it is not initialized.
        """
        if tick_end > tick_start:
            ticks = list(self._sorted_ticks.irange(tick_start + 1, tick_end))
            if not ticks or ticks[-1] != tick_end:
                ticks.append(tick_end)
        else:
            ticks = list(self._sorted_ticks.irange(tick_end, tick_start - 1, reverse=True))
            if not ticks or ticks[-1] != tick_end:
                ticks.append(tick_end)
        return ticks

    def cross_tick_up(self, tick: int) -> None:
        """Update active_liq when crossing tick upward."""
        self.active_liq += self.tick_deltas.get(tick, 0)

    def cross_tick_down(self, tick: int) -> None:
        """Update active_liq when crossing tick downward."""
        self.active_liq -= self.tick_deltas.get(tick, 0)

    def apply_swap(self, new_tick: int, new_sqrt_x96: int, reported_liq: int) -> None:
        """
        Update state after a swap. The pool reports its active_liq at the
        final tick; we use that as ground truth to correct any drift.
        """
        self.tick = new_tick
        self.sqrt_x96 = new_sqrt_x96
        self.active_liq = reported_liq
