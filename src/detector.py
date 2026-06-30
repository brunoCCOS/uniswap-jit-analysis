"""JIT sandwich detection within a single block."""

from dataclasses import dataclass, field


@dataclass
class JITSandwich:
    block_number: int
    owner: str
    mint_tx: str
    mint_tx_index: int
    burn_tx: str
    burn_tx_index: int
    tick_lower: int
    tick_upper: int
    jit_liquidity: int       # = mint.liquidity
    burn_liquidity: int      # may differ from mint for partial JIT
    jit_type: str            # "full" or "partial"
    new_passive_liq: int     # mint_liq - burn_liq (liq that stays as passive)
    swap_tx_hashes: list[str] = field(default_factory=list)


def detect_jit(rows: list[dict]) -> dict[str, JITSandwich]:
    """
    Detect JIT sandwiches in a single block's events (pre-sorted by tx_index, log_index).

    Returns a dict mapping each sandwiched swap_tx_hash → JITSandwich.
    Multiple swaps can map to the same sandwich (multi-swap JIT).
    """
    indexed = list(enumerate(rows))
    mints = [(i, r) for i, r in indexed if r["event"] == "mint"]
    burns = [(i, r) for i, r in indexed if r["event"] == "burn"]
    swaps = [(i, r) for i, r in indexed if r["event"] == "swap"]

    if not (mints and burns and swaps):
        return {}

    result: dict[str, JITSandwich] = {}
    used_burn_indices: set[int] = set()

    for mi, m in mints:
        owner = m.get("owner_address") or ""
        if not owner:
            continue
        t_lower = _int(m.get("tickLower"))
        t_upper = _int(m.get("tickUpper"))
        mint_liq = _int(m.get("liquidity"))
        if t_lower is None or t_upper is None or not mint_liq:
            continue
        mint_tx_idx = int(m["transaction_index"])

        for bi, b in burns:
            if bi in used_burn_indices:
                continue
            if b.get("owner_address") != owner:
                continue
            if bi <= mi:
                continue
            if _int(b.get("tickLower")) != t_lower or _int(b.get("tickUpper")) != t_upper:
                continue

            burn_tx_idx = int(b["transaction_index"])
            burn_liq = _int(b.get("liquidity")) or 0

            # Swaps that sit between mint and burn.
            # We do NOT filter by final tick here: a swap may traverse the JIT range
            # without ending inside it. Fee attribution per-segment handles that correctly.
            sandwiched = [
                r for _, r in swaps
                if mint_tx_idx < int(r["transaction_index"]) < burn_tx_idx
            ]
            if not sandwiched:
                continue

            jit_type = "full" if burn_liq >= mint_liq else "partial"
            new_passive = max(0, mint_liq - burn_liq)

            sandwich = JITSandwich(
                block_number=int(m["block_number"]),
                owner=owner,
                mint_tx=m["transaction_hash"],
                mint_tx_index=mint_tx_idx,
                burn_tx=b["transaction_hash"],
                burn_tx_index=burn_tx_idx,
                tick_lower=t_lower,
                tick_upper=t_upper,
                jit_liquidity=mint_liq,
                burn_liquidity=burn_liq,
                jit_type=jit_type,
                new_passive_liq=new_passive,
                swap_tx_hashes=[r["transaction_hash"] for r in sandwiched],
            )

            for r in sandwiched:
                result[r["transaction_hash"]] = sandwich

            used_burn_indices.add(bi)
            break  # greedy: match earliest valid burn per mint

    return result


def _int(v) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
        return int(f)
    except (ValueError, TypeError):
        return None


