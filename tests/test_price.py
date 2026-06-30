"""Tests for src/price.py — tick math, price conversion, segment amounts."""

import math
import pytest
from src.price import (
    Q96,
    Q192,
    parse_sqrt_x96,
    segment_amounts,
    sqrt_x96_to_price,
    tick_to_sqrt_x96,
)


# ── tick_to_sqrt_x96 ──────────────────────────────────────────────────────────

class TestTickToSqrtX96:
    def test_tick_zero_is_exactly_q96(self):
        assert tick_to_sqrt_x96(0) == Q96

    def test_monotonically_increasing(self):
        prev = tick_to_sqrt_x96(-10)
        for t in range(-9, 11):
            curr = tick_to_sqrt_x96(t)
            assert curr > prev, f"Not monotonic at tick {t}"
            prev = curr

    def test_positive_ticks_above_q96(self):
        for t in [1, 60, 10000, 100000]:
            assert tick_to_sqrt_x96(t) > Q96

    def test_negative_ticks_below_q96(self):
        for t in [-1, -60, -10000, -100000]:
            assert tick_to_sqrt_x96(t) < Q96

    def test_precision_vs_float_large_tick(self):
        """Decimal result should differ from float at large ticks (float loses bits)."""
        tick = 200000
        float_val = int(math.sqrt(1.0001 ** tick) * Q96)
        dec_val = tick_to_sqrt_x96(tick)
        assert dec_val != float_val, "Expected decimal to differ from float at tick=200000"

    def test_tick_1_satisfies_v3_ratio(self):
        """sqrt(1.0001^1) / Q96 should be ~1.00005."""
        val = tick_to_sqrt_x96(1)
        ratio = val / Q96
        assert abs(ratio - math.sqrt(1.0001)) < 1e-10

    def test_tick_60_satisfies_v3_ratio(self):
        val = tick_to_sqrt_x96(60)
        ratio = val / Q96
        assert abs(ratio - math.sqrt(1.0001 ** 60)) < 1e-10

    def test_inverse_symmetry(self):
        """tick_to_sqrt_x96(T) * tick_to_sqrt_x96(-T) ≈ Q192."""
        for t in [1, 100, 10000, 100000]:
            pos = tick_to_sqrt_x96(t)
            neg = tick_to_sqrt_x96(-t)
            product = pos * neg
            # Due to integer truncation the product won't be exactly Q192
            # but should be within 0.01% of it
            assert abs(product / Q192 - 1.0) < 1e-4, f"Symmetry broken at tick ±{t}"

    @pytest.mark.parametrize("tick,expected", [
        (0,      79228162514264337593543950336),
        (1,      79232123823359799118286999567),
        (-1,     79224201403219477170569942573),
        (60,     79466191966197645195421774832),
        (-60,    78990846045029531151608375685),
        (10000,  130621891405341611593710811005),
        (-10000, 48055510970269007215549348796),
    ])
    def test_known_values(self, tick, expected):
        result = tick_to_sqrt_x96(tick)
        # Allow ±1 from integer truncation
        assert abs(result - expected) <= 1, (
            f"tick={tick}: got {result}, expected {expected}"
        )


# ── sqrt_x96_to_price ─────────────────────────────────────────────────────────

class TestSqrtX96ToPrice:
    def test_zero_returns_zero(self):
        assert sqrt_x96_to_price(0, 6, 18) == 0.0

    def test_usdc_weth_known_swap(self):
        """Real swap: sqrtPriceX96 from pool, CSV shows WETH ≈ $2299."""
        sqrt_x96 = 1654623691489853604174858377438629
        price = sqrt_x96_to_price(sqrt_x96, 6, 18)
        # Allow 1% tolerance for float precision
        assert abs(price - 2299.17) / 2299.17 < 0.01

    def test_equal_decimals_gives_price_near_1(self):
        """For a stablecoin pool (6/6) at tick=0, price should be ~1."""
        price = sqrt_x96_to_price(Q96, 6, 6)
        assert abs(price - 1.0) < 1e-9

    def test_higher_tick_means_lower_usdc_per_weth(self):
        """For USDC(t0)/WETH(t1): higher tick → more WETH per USDC → LOWER USDC per WETH."""
        prices = [sqrt_x96_to_price(tick_to_sqrt_x96(t), 6, 18) for t in [190000, 200000, 210000]]
        assert prices[0] > prices[1] > prices[2], f"Expected decreasing USDC/WETH prices, got {prices}"

    def test_reciprocal_price(self):
        """1/price should give the inverse direction in human units."""
        sqrt_x96 = 1654623691489853604174858377438629
        usdc_per_weth = sqrt_x96_to_price(sqrt_x96, 6, 18)
        weth_per_usdc = 1.0 / usdc_per_weth
        # WETH per USDC ≈ 1/2299 ≈ 4.35e-4
        assert 4.0e-4 < weth_per_usdc < 4.7e-4

    def test_wbtc_usdc_price_range(self):
        """WBTC(8, t0)/USDC(6, t1): formula returns token0/token1 = WBTC per USDC."""
        import math as _m
        # price_raw = USDC_smallest / WBTC_smallest. At $30000 BTC:
        # 1 WBTC_smallest = 10^-8 BTC = $3e-4 = 300 USDC_smallest → price_raw = 300
        price_raw = 300.0
        sqrt_x96 = int(_m.sqrt(price_raw) * Q96)
        wbtc_per_usdc = sqrt_x96_to_price(sqrt_x96, 8, 6)
        # Should be ≈ 1/30000 ≈ 3.33e-5 WBTC per USDC
        assert abs(wbtc_per_usdc - 1 / 30000) / (1 / 30000) < 0.01, f"Got {wbtc_per_usdc}"


# ── parse_sqrt_x96 ────────────────────────────────────────────────────────────

class TestParseSqrtX96:
    def test_none_returns_zero(self):
        assert parse_sqrt_x96(None) == 0

    def test_empty_string_returns_zero(self):
        assert parse_sqrt_x96("") == 0

    def test_parses_small_int(self):
        assert parse_sqrt_x96("12345") == 12345

    def test_parses_large_int(self):
        val = "1654623691489853604174858377438629"
        assert parse_sqrt_x96(val) == 1654623691489853604174858377438629

    def test_invalid_string_returns_zero(self):
        assert parse_sqrt_x96("not_a_number") == 0


# ── segment_amounts ───────────────────────────────────────────────────────────

class TestSegmentAmounts:
    def test_zero_liquidity_returns_zeros(self):
        assert segment_amounts(Q96, Q96 + 10**20, 0, True) == (0, 0)

    def test_equal_sqrt_returns_zeros(self):
        assert segment_amounts(Q96, Q96, 10**15, True) == (0, 0)

    def test_direction_up_gives_positive_amounts(self):
        sqrt_a = tick_to_sqrt_x96(0)
        sqrt_b = tick_to_sqrt_x96(1)
        a0, a1 = segment_amounts(sqrt_a, sqrt_b, 10**15, True)
        assert a0 > 0 and a1 > 0

    def test_direction_down_gives_positive_amounts(self):
        sqrt_a = tick_to_sqrt_x96(1)
        sqrt_b = tick_to_sqrt_x96(0)
        a0, a1 = segment_amounts(sqrt_a, sqrt_b, 10**15, False)
        assert a0 > 0 and a1 > 0

    def test_known_amounts_tick_0_to_1(self):
        """Verify V3 formula: L * delta / Q96 and L * delta * Q96 / (sa * sb)."""
        sqrt_a = tick_to_sqrt_x96(0)   # = Q96
        sqrt_b = tick_to_sqrt_x96(1)
        L = 10 ** 15
        delta = sqrt_b - sqrt_a
        expected_a1 = L * delta // Q96
        expected_a0 = L * delta * Q96 // (sqrt_a * sqrt_b)
        a0, a1 = segment_amounts(sqrt_a, sqrt_b, L, True)
        assert a0 == expected_a0
        assert a1 == expected_a1

    def test_amounts_scale_linearly_with_liquidity(self):
        sqrt_a = tick_to_sqrt_x96(0)
        sqrt_b = tick_to_sqrt_x96(60)
        a0_1x, a1_1x = segment_amounts(sqrt_a, sqrt_b, 10**12, True)
        a0_2x, a1_2x = segment_amounts(sqrt_a, sqrt_b, 2 * 10**12, True)
        # Allow ±1 for integer floor-division rounding: 2*(a//n) may differ from (2a)//n by 1
        assert abs(a0_2x - 2 * a0_1x) <= 1
        assert abs(a1_2x - 2 * a1_1x) <= 1

    def test_symmetric_amounts_for_both_directions(self):
        """The segment amounts only depend on |delta|, not direction."""
        sqrt_a = tick_to_sqrt_x96(100)
        sqrt_b = tick_to_sqrt_x96(160)
        L = 5 * 10**14
        a0_up, a1_up = segment_amounts(sqrt_a, sqrt_b, L, True)
        a0_dn, a1_dn = segment_amounts(sqrt_b, sqrt_a, L, False)
        assert a0_up == a0_dn
        assert a1_up == a1_dn

    def test_amount_ratio_matches_sqrt_price_product(self):
        """V3 identity: amount1 / amount0 ≈ sqrt_a * sqrt_b / Q96^2."""
        sqrt_a = tick_to_sqrt_x96(100)
        sqrt_b = tick_to_sqrt_x96(200)
        L = 10**18
        a0, a1 = segment_amounts(sqrt_a, sqrt_b, L, True)
        assert a0 > 0 and a1 > 0
        # ratio: a1 / a0 = (delta/Q96) / (delta*Q96/(sa*sb)) = sa*sb/Q96^2
        expected_ratio = sqrt_a * sqrt_b / Q96 ** 2
        actual_ratio = a1 / a0
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-6

    def test_large_tick_range(self):
        """Amounts should be positive for a wide tick range."""
        sqrt_a = tick_to_sqrt_x96(-100000)
        sqrt_b = tick_to_sqrt_x96(100000)
        a0, a1 = segment_amounts(sqrt_a, sqrt_b, 10**12, True)
        assert a0 > 0 and a1 > 0
