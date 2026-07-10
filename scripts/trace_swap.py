"""
Full step-by-step trace of BOTH optimizers for a single (problematic) swap.

Captures the swap's inputs from a live pipeline run, then replays the analytical
and combinatorial optimizers with detailed per-step logging so the two can be
compared against the paper's expected behavior.

Usage: python -m scripts.trace_swap <pool_id> <initial_tick>
"""

import sys
import os
import contextlib
import io

sys.path.insert(0, os.path.expanduser("~/JITUniswapOptimization"))

import src.optimizers as O
import src.enricher as E
from uniswap_utils.position import Position
from uniswap_utils.utils import sqrt_price_from_tick, tick_from_sqrt_price, get_rounded_tick
from optimization.analytical import AnalyticalOptimizer
from optimization.search import ternary_search_max


def _run_inline_trace(pool_id: str, target_tick: int) -> None:
    """Run the pipeline; when the target swap is reached, trace it INLINE.

    The PoolState is mutable and shared across swaps, so it must be traced
    during the live optimizer call (not captured for later use, when its price
    has already advanced to subsequent swaps).
    """
    from src.config import POOL_BY_ID
    from scripts.process import run_pool
    cfg = POOL_BY_ID[pool_id]
    orig = O.run_analytical_optimizer
    done = {"x": False}

    def cap(**kw):
        if kw["initial_tick"] == target_tick and not done["x"]:
            done["x"] = True
            util, actual, liq_scale = O._build_jit_objects(
                kw["state"], kw["jit"], cfg, kw["amount_in_raw"],
                kw["direction_up"], kw["p0"], kw["p1"], kw["active_liq_start"],
            )
            import sys as _sys
            out = _real_stdout
            print(f"POOL {pool_id}  SWAP initial_tick={target_tick}  budget=${kw['budget']:.2f}", file=out)
            print(f"jit on-chain: range=[{kw['jit'].tick_lower},{kw['jit'].tick_upper}] "
                  f"L={kw['jit'].jit_liquidity:.4e}  direction_up={kw['direction_up']}", file=out)
            with contextlib.redirect_stdout(out):
                trace_analytical(util, kw["budget"], kw["direction_up"], cfg)
                trace_combinatorial(util, kw["budget"], cfg)
        return orig(**kw)

    O.run_analytical_optimizer = cap
    E.run_analytical_optimizer = cap
    with contextlib.redirect_stdout(io.StringIO()):
        run_pool(cfg)
    O.run_analytical_optimizer = orig
    E.run_analytical_optimizer = orig
    if not done["x"]:
        raise SystemExit(f"swap with initial_tick={target_tick} not found in {pool_id}")


def trace_analytical(util, budget, direction_up, cfg):
    print("\n" + "=" * 78)
    print("ANALYTICAL (closed-form) — step by step")
    print("=" * 78)
    st = util.swap.state
    ts = st.tick_space
    dec0, dec1 = st.dec0, st.dec1
    F = 1.0 + float(st.fee_rate)
    Delta_x = float(util.swap.amount_in)
    pool_sqrt = float(st.price)

    if direction_up:
        px, py = float(util.price1), float(util.price0)
        init_sqrt = 1.0 / pool_sqrt
        canon = lambda t: 1.0 / float(sqrt_price_from_tick(t, dec0, dec1))
    else:
        px, py = float(util.price0), float(util.price1)
        init_sqrt = pool_sqrt
        canon = lambda t: float(sqrt_price_from_tick(t, dec0, dec1))

    B_tokens = budget / py
    cur = tick_from_sqrt_price(st.price, dec0, dec1)
    start, _ = get_rounded_tick(cur, ts)
    end = util.swap.simulate(Position(0, 0, 0))["final_tick"]
    print(f"direction_up={direction_up}  px={px:.6f} py={py:.6f}  F={F}")
    print(f"pool_sqrt={pool_sqrt:.8f}  init_sqrt(canonical)={init_sqrt:.8f}")
    print(f"gross amount_in={Delta_x:.6e}  net_total=Delta_x/F={Delta_x/F:.6e}")
    print(f"budget=${budget:.2f}  B_tokens=budget/py={B_tokens:.6e}")
    print(f"current_tick={cur}  start_tick={start}  no-JIT end_tick={end}")

    opt = AnalyticalOptimizer(util.swap, util.price0, util.price1)
    ranges = opt._build_ranges(start, end, ts, direction_up)
    if not ranges:
        ranges = [(start, start + ts)]
    print(f"candidate ranges (j=0..): {ranges}")

    params = opt._precompute(ranges, Delta_x, B_tokens, px, py, F, init_sqrt, canon)
    print("\nper-range precompute + lemma + BOTH utilities:")
    for j, p in enumerate(params):
        print(f"\n  j={j}  range=[{p.lower},{p.upper}]")
        print(f"    P={p.P:.6e}  dx={p.dx:.6e}  R={p.R:.8f}  C={p.C:.6e}")
        print(f"    A={p.A:.8f}  L_inner={p.L_inner:.6e}  L0={p.L0:.6e}  L_max={p.L_max:.6e}")
        print(f"    cap_per_L={p.cap_per_L:.8e}  traversed_cap={p.traversed_cap:.8e}")
        print(f"    feasible containment? L0<=L_max : {p.L0 <= p.L_max}")
        if j == 0:
            L = opt._lemma_5_1(p, F)
            print(f"    lemma_5_1 -> L={L:.6e}")
            tgt = p
        else:
            prev = params[j - 1]
            L_low, L_up = opt._lemma_5_2(p, prev, F, px)
            tgt, L = (prev, L_up) if L_up > 0 else (p, L_low)
            print(f"    lemma_5_2 -> L_low={L_low:.6e} L_up={L_up:.6e} "
                  f"-> target=[{tgt.lower},{tgt.upper}] L={L:.6e}")
        cf = opt._utility(L, tgt.P, tgt.dx, tgt.R, tgt.C, F, px)
        sim = util.position_utility(tgt.lower, tgt.upper, L)
        print(f"    closed-form utility = {cf:.4f}")
        print(f"    SIMULATION utility  = {sim:.4f}   (diff={cf - sim:.4f})")

    res = opt.optimize(budget) if hasattr(opt, "optimize") else None
    print(f"\n  ANALYTICAL PICKS: {res}")
    if res and res.get("lower_tick") is not None:
        print(f"  its simulated utility = "
              f"{util.position_utility(res['lower_tick'], res['upper_tick'], res['liquidity']):.4f}")


def trace_combinatorial(util, budget, cfg):
    print("\n" + "=" * 78)
    print("COMBINATORIAL (simulation) — step by step")
    print("=" * 78)
    st = util.swap.state
    ts = st.tick_space
    end = util.swap.simulate(Position(0, 0, 0))["final_tick"]
    cur = tick_from_sqrt_price(st.price, st.dec0, st.dec1)
    start, _ = get_rounded_tick(cur, ts)
    end_r = (end // ts) * ts
    lo, hi = min(start, end_r), max(start, end_r)
    print(f"current_tick={cur} start_tick={start} no-JIT end_tick={end} -> enum [{lo},{hi}]")
    best = (-1e18, None)
    print("\nper-candidate-range line search (simulation):")
    for a in range(lo, hi + ts, ts):
        for b in range(a + ts, hi + 2 * ts, ts):
            util.set_ticks(a, b)
            max_liq = Position(0, a, b).liqudity_from_budget(
                budget, st.price, util.price0, util.price1, st.dec0, st.dec1
            )
            opt_u, opt_L = ternary_search_max(util.utility_liq, 0, max_liq)
            pv = Position(opt_L, a, b).value(st.price, util.price0, util.price1, st.dec0, st.dec1)
            flag = ""
            if opt_u > best[0]:
                best = (opt_u, (a, b, opt_L))
                flag = "  <- best so far"
            print(f"  [{a},{b}] max_liq={max_liq:.4e} opt_L={opt_L:.4e} "
                  f"sim_util={opt_u:.4f} pos_value=${pv:.2f}{flag}")
    print(f"\n  COMBINATORIAL PICKS: range={best[1][:2]} L={best[1][2]:.4e} util={best[0]:.4f}")


_real_stdout = sys.stdout


def main():
    pool_id = sys.argv[1] if len(sys.argv) > 1 else "2697647"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 62697
    _run_inline_trace(pool_id, target)


if __name__ == "__main__":
    main()
