from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("/home/brunollacer/uniswap")

# Known ERC-20 token decimals by address (lowercase)
TOKEN_DECIMALS: dict[str, int] = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
}

TOKEN_SYMBOLS: dict[str, str] = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
}

TICK_SPACING_TO_FEE: dict[int, int] = {
    1: 100,
    10: 500,
    60: 3000,
    200: 10000,
}


@dataclass(frozen=True)
class PoolConfig:
    pool_id: str
    pool_dir: Path
    token0_address: str
    token1_address: str
    tick_spacing: int
    fee_millionths: int  # e.g. 3000 for 0.3%

    @property
    def token0_decimals(self) -> int:
        return TOKEN_DECIMALS[self.token0_address.lower()]

    @property
    def token1_decimals(self) -> int:
        return TOKEN_DECIMALS[self.token1_address.lower()]

    @property
    def token0_symbol(self) -> str:
        return TOKEN_SYMBOLS.get(self.token0_address.lower(), "TOKEN0")

    @property
    def token1_symbol(self) -> str:
        return TOKEN_SYMBOLS.get(self.token1_address.lower(), "TOKEN1")

    @property
    def total_csv(self) -> Path:
        return self.pool_dir / f"{self.pool_id}-Total.csv"

    @property
    def pair_label(self) -> str:
        return f"{self.token0_symbol}/{self.token1_symbol}"


# fmt: off
ALL_POOLS: list[PoolConfig] = [
    PoolConfig(
        pool_id="2697585",
        pool_dir=DATA_ROOT / "2697585-eth-usdc-fee-100",
        token0_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        token1_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        tick_spacing=200,
        fee_millionths=10000,
    ),
    PoolConfig(
        pool_id="2697588",
        pool_dir=DATA_ROOT / "2697588-usdt-usdc-fee-5",
        token0_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        token1_address="0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        tick_spacing=10,
        fee_millionths=500,
    ),
    PoolConfig(
        pool_id="2697600",
        pool_dir=DATA_ROOT / "2697600-eth-usdc-fee-30",
        token0_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        token1_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        tick_spacing=60,
        fee_millionths=3000,
    ),
    PoolConfig(
        pool_id="2697647",
        pool_dir=DATA_ROOT / "2697647-wbtc-usdc-fee-30",
        token0_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
        token1_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        tick_spacing=60,
        fee_millionths=3000,
    ),
    PoolConfig(
        pool_id="2697765",
        pool_dir=DATA_ROOT / "2697765-eth-usdc-fee-5",
        token0_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        token1_address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        tick_spacing=10,
        fee_millionths=500,
    ),
]
# fmt: on

POOL_BY_ID: dict[str, PoolConfig] = {p.pool_id: p for p in ALL_POOLS}
