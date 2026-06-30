"""CSV loading with explicit schema to avoid Polars type inference overhead."""

import polars as pl

from src.config import PoolConfig

# Columns we actually use; rest are dropped after load to save memory
USED_COLUMNS = [
    "block_number",
    "timestamp",
    "log_index",
    "transaction_index",
    "transaction_hash",
    "event",
    "owner_address",
    "txFrom",
    "sender_address",
    "recipient_address",
    "amount0",
    "token0_price_usd",
    "amount1",
    "token1_price_usd",
    "liquidity",
    "sqrtPriceX96",
    "tick",
    "tickLower",
    "tickUpper",
]

CSV_SCHEMA_OVERRIDES = {
    "block_number": pl.Int64,
    "log_index": pl.Int32,
    "transaction_index": pl.Int32,
    "liquidity": pl.Float64,   # use Float64 to handle nulls cleanly; convert to int in state machine
    "sqrtPriceX96": pl.Utf8,   # >64-bit, keep as string
    "tick": pl.Float64,         # nullable (missing for mint/burn)
    "tickLower": pl.Float64,    # nullable
    "tickUpper": pl.Float64,    # nullable
    "amount0": pl.Float64,
    "amount1": pl.Float64,
    "token0_price_usd": pl.Float64,
    "token1_price_usd": pl.Float64,
}


def load_events(cfg: PoolConfig) -> pl.DataFrame:
    """
    Load and sort pool events. Returns a DataFrame with only the columns we need,
    sorted by (block_number, transaction_index, log_index).
    """
    df = (
        pl.scan_csv(
            cfg.total_csv,
            schema_overrides=CSV_SCHEMA_OVERRIDES,
            null_values=["", "NA", "NaN"],
            truncate_ragged_lines=True,
        )
        .select(USED_COLUMNS)
        .sort(["block_number", "transaction_index", "log_index"])
        .collect()
    )
    return df
