"""
Bridge between the uniswap-jit-analysis PoolState and the JITUniswapOptimization
library.  Provides two public functions that build the cross-repo objects, run the
two optimizers, and score the actual on-chain JIT position via simulation.

Unit system note
----------------
The JITUniswapOptimization library uses human-unit sqrt prices
(sqrt_price_from_tick gives ~0.02 for USDC/WETH rather than the raw Q64.96 value).
Because of this, its liquidity unit ("lib L") is related to raw Uniswap L by:

    L_lib = L_raw / 10^((dec0 + dec1) / 2)

For USDC(6)/WETH(18): L_lib = L_raw / 1e12.
For WBTC(8)/USDC(6):  L_lib = L_raw / 1e7.

Amounts in input tokens follow the same logic:

    amount_in_lib = amount_in_raw / 10^dec_in

All conversions are done once in _build_jit_objects; the library sees internally
consistent numbers and the returned liquidity is converted back to raw Uniswap L
so that it is comparable to the on-chain jit_liquidity column.
"""

from __future__ import annotations

import os
import sys

# Make the JITUniswapOptimization library importable without installing it.
_JIT_REPO = os.path.expanduser("~/JITUniswapOptimization")
if _JIT_REPO not in sys.path:
    sys.path.insert(0, _JIT_REPO)

from uniswap_utils.state import State  # noqa: E402  (after sys.path mutation)
from uniswap_utils.swap import Swap  # noqa: E402
from optimization.utility import Utility  # noqa: E402

from src.detector import JITSandwich
from src.config import PoolConfig
from src.state import PoolState

Q96 = 2**96


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sqrt_human(pool_state: PoolState, dec0: int, dec1: int) -> float:
    """Convert Q64.96 sqrtPriceX96 to the human-unit sqrt price used by the lib."""
    return pool_state.sqrt_x96 / Q96 * 10 ** ((dec0 - dec1) / 2)


def _build_passive_dict(
    pool_state: PoolState, jit: JITSandwich, tick_spacing: int, active_liq_start: int
) -> dict[int, int]:
    """
    Reconstruct {lower_tick → passive_liquidity_in_range} in raw Uniswap L units.

    active_liq_start is the pool's active liquidity at the swap's initial tick,
    captured before the segment walk mutated pool_state.active_liq via tick
    crossings.  pool_state.active_liq at this point holds the *final*-tick value
    (post-walk), which anchors the profile to the wrong tick and, once the JIT is
    subtracted, can wrongly clamp the passive book to zero.  It includes the JIT
    position (minted before the swap), so we subtract it to get passive-only.
    """
    start_lower = (pool_state.tick // tick_spacing) * tick_spacing

    base_liq = active_liq_start
    if jit.tick_lower <= pool_state.tick < jit.tick_upper:
        base_liq = max(0, base_liq - jit.jit_liquidity)

    def _passive_delta(t: int) -> int:
        """Net tick delta with JIT's own contribution removed."""
        raw = pool_state.tick_deltas.get(t, 0)
        if t == jit.tick_lower:
            raw -= jit.jit_liquidity
        if t == jit.tick_upper:
            raw += jit.jit_liquidity
        return raw

    result: dict[int, int] = {start_lower: base_liq}

    # Walk upward: crossing tick t adds delta[t] to liquidity.
    liq = base_liq
    for t in sorted(t for t in pool_state.tick_deltas if t > start_lower):
        liq = max(0, liq + _passive_delta(t))
        result[t] = liq

    # Walk downward: uncrossing tick t subtracts delta[t].
    # result[t] is the range [t, t+ts) BEFORE uncrossing t (= liq from the
    # range above), and result[t-ts] is the range [t-ts, t) AFTER uncrossing.
    # Without the result[t] assignment, ranges between consecutive initialized
    # ticks that lie below start_lower would be missing from the dict.
    liq = base_liq
    for t in sorted((t for t in pool_state.tick_deltas if t <= start_lower), reverse=True):
        result[t] = liq
        liq = max(0, liq - _passive_delta(t))
        result[t - tick_spacing] = liq

    # The library simulator reads this profile with an exact per-tick lookup and
    # steps one tick_spacing at a time, so every tick multiple the swap can visit
    # needs its own entry. `result` only has entries at initialized ticks (where
    # liquidity changes); liquidity is constant from each initialized tick up to
    # the next. Densify by filling every tick_spacing multiple in each
    # constant-liquidity span with that span's value. Empty spans (liquidity 0)
    # are omitted so a swap that reaches a genuinely liquidity-less region still
    # stops there rather than reading a stale neighbouring value.
    dense: dict[int, int] = {}
    ticks = sorted(result)
    for i, t in enumerate(ticks):
        liq = result[t]
        if liq <= 0:
            continue
        next_t = ticks[i + 1] if i + 1 < len(ticks) else t + tick_spacing
        for cur in range(t, next_t, tick_spacing):
            dense[cur] = liq
    return dense


def _build_jit_objects(
    pool_state: PoolState,
    jit: JITSandwich,
    cfg: PoolConfig,
    amount_in_raw: float,
    direction_up: bool,
    p0: float,
    p1: float,
    active_liq_start: int,
) -> tuple[Utility, float, float]:
    """
    Construct the cross-repo Swap + Utility objects with proper unit scaling.

    Returns (utility, actual_utility_usd, liq_scale) where:
    - actual_utility_usd is the simulation P&L of the actual on-chain JIT position
    - liq_scale = 10^((dec0+dec1)/2) converts library L back to raw Uniswap L
    """
    dec0, dec1 = cfg.token0_decimals, cfg.token1_decimals

    # Scale factor between raw Uniswap L and library L (see module docstring).
    liq_scale = 10 ** ((dec0 + dec1) / 2)

    # Build passive dict in raw units, then convert to library units.
    passive_dict_raw = _build_passive_dict(
        pool_state, jit, cfg.tick_spacing, active_liq_start
    )
    passive_dict_lib = {k: float(v) / liq_scale for k, v in passive_dict_raw.items()}

    # Amount in library units (human token amounts).
    dec_in = dec1 if direction_up else dec0
    amount_in_lib = amount_in_raw / 10**dec_in

    jit_state = State(
        price=_sqrt_human(pool_state, dec0, dec1),
        passive_dict=passive_dict_lib,
        tick_space=cfg.tick_spacing,
        fee_rate=cfg.fee_millionths / 1_000_000,
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(
        amount_in=amount_in_lib,
        zeroForOne=not direction_up,
        state=jit_state,
    )
    utility = Utility(swap, p0, p1)

    # Actual on-chain JIT position in library L units.
    jit_liq_lib = jit.jit_liquidity / liq_scale
    actual_util = float(utility.position_utility(
        jit.tick_lower, jit.tick_upper, jit_liq_lib
    ))
    return utility, actual_util, liq_scale


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def run_simulation_optimizer(
    state: PoolState,
    initial_sqrt: int,
    initial_tick: int,
    active_liq_start: int,
    jit: JITSandwich,
    cfg: PoolConfig,
    p0: float,
    p1: float,
    direction_up: bool,
    amount_in_raw: float,
    no_jit_final_tick: int,
    budget: float,
) -> dict | None:
    """
    Combinatorial (simulation-based) optimizer.

    Returns a dict with keys matching the enricher's `optimal_*` columns plus
    `actual_utility_usd`.  Returns None on any error.
    """
    try:
        utility, actual_util, liq_scale = _build_jit_objects(
            state, jit, cfg, amount_in_raw, direction_up, p0, p1, active_liq_start
        )
        result = utility.optimize(
            budget,
            method="combinatorial",
        )
        if result.get("lower_tick") is None:
            return None
        optimal_util = float(
            utility.position_utility(
                result["lower_tick"], result["upper_tick"], result["liquidity"]
            )
        )
        return {
            "optimal_tick_lower": result["lower_tick"],
            "optimal_tick_upper": result["upper_tick"],
            # Convert library L back to raw Uniswap L for comparability.
            "optimal_jit_liquidity": float(result["liquidity"]) * liq_scale,
            "optimal_utility_usd": optimal_util,
            "actual_utility_usd": actual_util,
        }
    except Exception:
        return None


def run_analytical_optimizer(
    state: PoolState,
    initial_sqrt: int,
    initial_tick: int,
    active_liq_start: int,
    jit: JITSandwich,
    cfg: PoolConfig,
    p0: float,
    p1: float,
    direction_up: bool,
    amount_in_raw: float,
    no_jit_final_tick: int,
    budget: float,
) -> dict | None:
    """
    Analytical (closed-form Lemma 5.1/5.2) optimizer.

    Returns a dict with keys matching the enricher's `kh_*` columns plus
    `kh_actual_utility_usd`.  Returns None on any error.
    """
    try:
        utility, actual_util, liq_scale = _build_jit_objects(
            state, jit, cfg, amount_in_raw, direction_up, p0, p1, active_liq_start
        )
        result = utility.optimize(budget, method="analytical")
        if result.get("lower_tick") is None:
            return None
        optimal_util = float(
            utility.position_utility(
                result["lower_tick"], result["upper_tick"], result["liquidity"]
            )
        )
        return {
            "kh_tick_lower": result["lower_tick"],
            "kh_tick_upper": result["upper_tick"],
            "kh_jit_liquidity": float(result["liquidity"]) * liq_scale,
            "kh_optimal_utility_usd": optimal_util,
            "kh_actual_utility_usd": actual_util,
        }
    except Exception:
        return None
