"""sqrtPriceX96 ↔ human-readable price conversions."""

import math

Q96 = 1 << 96
Q192 = 1 << 192


def tick_to_sqrt_x96(tick: int) -> int:
    """Return sqrtPriceX96 for a given tick boundary."""
    return int(math.sqrt(1.0001**tick) * Q96)


def sqrt_x96_to_price(sqrt_x96: int, token0_decimals: int, token1_decimals: int) -> float:
    """
    Convert sqrtPriceX96 to a human-readable price:
        price = (token0 per token1) adjusted for decimals

    For USDC(6)/WETH(18): price_raw = (sqrt/Q96)^2 * 10^(18-6) → USDC per WETH.
    """
    if sqrt_x96 == 0:
        return 0.0
    price_raw = (sqrt_x96 / Q96) ** 2
    return price_raw * (10 ** (token1_decimals - token0_decimals))


def parse_sqrt_x96(value: str | None) -> int:
    """Parse a sqrtPriceX96 string to a Python int (handles big integers)."""
    if not value:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def segment_amounts(
    sqrt_a: int,
    sqrt_b: int,
    liquidity: int,
    direction_up: bool,
) -> tuple[int, int]:
    """
    Compute token amounts for a constant-liquidity tick segment.
    Returns (amount0, amount1) where the input token is positive.

    direction_up=True: sqrtPrice moves from sqrt_a to sqrt_b (sqrt_b > sqrt_a)
        → token1 flows in, token0 flows out
        amount1_in  = L * (sqrt_b - sqrt_a) / Q96
        amount0_out = L * (sqrt_b - sqrt_a) * Q96 / (sqrt_a * sqrt_b)

    direction_up=False: sqrtPrice moves from sqrt_a to sqrt_b (sqrt_b < sqrt_a)
        → token0 flows in, token1 flows out
        amount0_in  = L * (sqrt_a - sqrt_b) * Q96 / (sqrt_a * sqrt_b)
        amount1_out = L * (sqrt_a - sqrt_b) / Q96
    """
    if sqrt_a == sqrt_b or liquidity == 0:
        return 0, 0

    if direction_up:
        delta = sqrt_b - sqrt_a
        amount1 = liquidity * delta // Q96
        # amount0 is output (negative), but return as positive for volume calc
        amount0 = liquidity * delta * Q96 // (sqrt_a * sqrt_b) if sqrt_a and sqrt_b else 0
        return amount0, amount1
    else:
        delta = sqrt_a - sqrt_b
        amount0 = liquidity * delta * Q96 // (sqrt_a * sqrt_b) if sqrt_a and sqrt_b else 0
        amount1 = liquidity * delta // Q96
        return amount0, amount1
