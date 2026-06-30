"""sqrtPriceX96 ↔ human-readable price conversions."""

import decimal

Q96 = 1 << 96
Q192 = 1 << 192

_DEC_PREC = 50
_BASE = decimal.Decimal("1.0001")
_Q96_DEC = decimal.Decimal(Q96)


def tick_to_sqrt_x96(tick: int) -> int:
    """Return sqrtPriceX96 for a given tick boundary (high-precision decimal math)."""
    with decimal.localcontext() as ctx:
        ctx.prec = _DEC_PREC
        ratio = _BASE ** decimal.Decimal(tick)
        sqrt_ratio = ratio.sqrt()
        return int(sqrt_ratio * _Q96_DEC)


def sqrt_x96_to_price(sqrt_x96: int, token0_decimals: int, token1_decimals: int) -> float:
    """
    Convert sqrtPriceX96 to human-readable token1 price in token0 units.

    sqrtPriceX96^2 / Q192 = token1_raw / token0_raw (V3 convention).
    Inverting and adjusting for decimals gives token0 per token1 in human units:
        price = (Q96 / sqrt_x96)^2 * 10^(token1_decimals - token0_decimals)

    For USDC(6)/WETH(18): returns USDC per WETH (~2000–4000 range).
    """
    if sqrt_x96 == 0:
        return 0.0
    price_raw = (sqrt_x96 / Q96) ** 2
    if price_raw == 0.0:
        return 0.0
    return (10 ** (token1_decimals - token0_decimals)) / price_raw


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
