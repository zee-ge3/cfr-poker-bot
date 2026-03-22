"""
Offline script: compute preflop CFR+ strategies for 169 hand buckets.

Saves result to nlhe/cfr/tables/preflop_strategy.npy
  shape: (169, 2, 5)  — bucket x position x action
  dtype: float32

Game model:
  2 players, starting stack 100, SB posts 1, BB posts 2.
  5 preflop actions: FOLD(0), CALL(1), RAISE_SMALL(2), RAISE_LARGE(3), RAISE_ALLIN(4).
  Terminal value based on hand equity vs uniform range.

Run:
  python -m nlhe.cfr.preflop_compute
"""

import os
import numpy as np

from nlhe.cfr.abstraction import (
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN,
    PREFLOP_N_ACTIONS,
)

N_ACTIONS      = PREFLOP_N_ACTIONS   # 5
STARTING_STACK = 100.0
SB_POST        = 1.0
BB_POST        = 2.0

# ---------------------------------------------------------------------------
# Bucket equity table
# ---------------------------------------------------------------------------

def _bucket_equity(bucket: int) -> float:
    """
    Approximate heads-up equity vs uniform random range.
      0-12  : pairs 22=0 .. AA=12 -> 0.52 + (b/12)*0.33
      13-90 : suited AKs=13 .. 32s=90 -> 0.67 - ((b-13)/77)*0.32
      91-168: offsuit AKo=91 .. 32o=168 -> 0.65 - ((b-91)/77)*0.32
    """
    if bucket <= 12:
        return 0.52 + (bucket / 12.0) * 0.33
    elif bucket <= 90:
        return 0.67 - ((bucket - 13) / 77.0) * 0.32
    else:
        return 0.65 - ((bucket - 91) / 77.0) * 0.32


BUCKET_EQUITY = np.array([_bucket_equity(b) for b in range(169)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Explicit preflop betting tree
# ---------------------------------------------------------------------------
# We model the heads-up preflop hand as a finite, two-street (one per player)
# betting tree with a depth limit of one raise each.
#
# Raise sizing (applied to the current pot):
#   RAISE_SMALL  : 0.33 × pot (minimum 3 chips above previous bet)
#   RAISE_LARGE  : 0.75 × pot
#   RAISE_ALLIN  : full remaining stack
#
# Terminal EV for position p at showdown with equity e and pot P:
#   EV_p = e * P  - (chips_p_invested)
# where chips_p_invested = (starting_stack - remaining_stack_p).
#
# We track SB and BB stacks.  All EVs are returned from SB's perspective
# (positive = good for SB).

def _raise_size(action: int, pot: float, stack: float) -> float:
    """Concrete raise size in chips for a preflop action."""
    if action == RAISE_ALLIN:
        return stack
    elif action == RAISE_SMALL:
        return min(pot * 0.33, stack)
    elif action == RAISE_LARGE:
        return min(pot * 0.75, stack)
    else:
        raise ValueError(f"Not a raise: {action}")


def _showdown_ev(equity: float, pot: float,
                 stack_sb: float, stack_bb: float) -> float:
    """SB's EV at showdown."""
    invested_sb = STARTING_STACK - stack_sb
    return equity * pot - invested_sb


def _fold_ev(pot: float, stack_sb: float, stack_bb: float,
             folder: int) -> float:
    """SB's EV when `folder` folds (0=SB folds, 1=BB folds)."""
    if folder == 0:
        # SB folds — SB loses their investment
        return -(STARTING_STACK - stack_sb)
    else:
        # BB folds — SB wins the pot
        return STARTING_STACK - stack_sb  # net = pot - invested_sb = +invested_bb


# ---------------------------------------------------------------------------
# Node-based CFR+ with explicit game tree
# ---------------------------------------------------------------------------
#
# Nodes:
#   0: SB acts first  (pot=3, SB faces 1 to call BB's extra chip)
#   1: BB responds to SB's raise (pot varies)
#   2: SB responds to BB's re-raise (pot varies)
#
# Each node is solved per-bucket independently.

class _PreflopCFR:
    """Per-bucket CFR+ solver over a depth-limited 3-node tree."""

    def __init__(self, equity: float):
        self.equity = equity
        # Regrets and strategy sums for 3 nodes × 5 actions
        self.regret    = np.zeros((3, N_ACTIONS), dtype=np.float64)
        self.strat_sum = np.zeros((3, N_ACTIONS), dtype=np.float64)

    def _strategy(self, node: int) -> np.ndarray:
        """Current strategy at node (regret-matching, CFR+ floors at 0)."""
        reg = np.maximum(self.regret[node], 0.0)
        s = reg.sum()
        if s > 0:
            return reg / s
        return np.ones(N_ACTIONS, dtype=np.float64) / N_ACTIONS

    # --- Node 0: SB acts (pot=3, SB has 99, BB has 98, SB must call 1) ----

    def _node0(self, t: int, sb: float, bb: float, pot: float) -> float:
        """SB's decision. Returns SB EV."""
        strat = self._strategy(0)
        self.strat_sum[0] += (t + 1) * strat

        ev_actions = np.zeros(N_ACTIONS, dtype=np.float64)

        # FOLD (SB)
        ev_actions[FOLD] = _fold_ev(pot, sb, bb, folder=0)

        # CALL (SB matches BB's 1-chip advantage → invests 1 more)
        call_chips = min(1.0, sb)   # SB needs 1 more chip to call BB=2
        new_sb  = sb - call_chips
        new_pot = pot + call_chips
        ev_actions[CALL] = _showdown_ev(self.equity, new_pot, new_sb, bb)

        # RAISE actions (SB raises into BB)
        for a in (RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN):
            raise_chips = _raise_size(a, pot, sb)
            # SB raises: SB adds raise_chips to pot
            new_sb2  = sb - raise_chips
            new_pot2 = pot + raise_chips
            # BB responds at node 1
            ev_actions[a] = self._node1(t, new_sb2, bb, new_pot2, raise_chips)

        node_ev = float(strat @ ev_actions)

        # Update regrets (CFR+: floor at 0)
        self.regret[0] = np.maximum(0.0, self.regret[0] + ev_actions - node_ev)

        return node_ev

    # --- Node 1: BB responds to SB's raise --------------------------------

    def _node1(self, t: int, sb: float, bb: float, pot: float,
                facing: float) -> float:
        """BB's decision after SB raises. Returns SB EV."""
        strat = self._strategy(1)
        self.strat_sum[1] += (t + 1) * strat

        ev_actions = np.zeros(N_ACTIONS, dtype=np.float64)

        # FOLD (BB)
        ev_actions[FOLD] = _fold_ev(pot, sb, bb, folder=1)

        # CALL (BB matches SB's raise — BB must put in `facing` more chips)
        call_chips = min(facing, bb)
        new_bb   = bb - call_chips
        new_pot2 = pot + call_chips
        ev_actions[CALL] = _showdown_ev(self.equity, new_pot2, sb, new_bb)

        # RE-RAISE actions (BB re-raises)
        for a in (RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN):
            reraise_chips = _raise_size(a, pot, bb)
            # BB must first call the facing bet, then raise on top
            total_bb_invest = min(reraise_chips, bb)
            new_bb2   = bb - total_bb_invest
            new_pot3  = pot + total_bb_invest
            # SB responds at node 2
            ev_actions[a] = self._node2(t, sb, new_bb2, new_pot3, total_bb_invest)

        node_ev = float(strat @ ev_actions)
        # Note: node_ev is SB's EV; BB's EV = −node_ev
        # Regrets are from BB's perspective (negative of SB's)
        bb_ev_actions = -ev_actions
        bb_node_ev    = -node_ev
        self.regret[1] = np.maximum(0.0, self.regret[1] + bb_ev_actions - bb_node_ev)

        return node_ev

    # --- Node 2: SB responds to BB's re-raise (depth limit: call or fold) --

    def _node2(self, t: int, sb: float, bb: float, pot: float,
                facing: float) -> float:
        """SB's decision after BB re-raises. Only FOLD or CALL (no further raises)."""
        # Only 2 meaningful actions at depth limit
        strat2 = self._strategy(2)
        # Mask: only FOLD and CALL matter here
        strat_fc = np.zeros(N_ACTIONS, dtype=np.float64)
        strat_fc[FOLD] = strat2[FOLD]
        strat_fc[CALL] = 1.0 - strat2[FOLD]
        self.strat_sum[2] += (t + 1) * strat_fc

        ev_fold = _fold_ev(pot, sb, bb, folder=0)
        call_chips = min(facing, sb)
        ev_call = _showdown_ev(self.equity, pot + call_chips, sb - call_chips, bb)

        ev_actions = np.zeros(N_ACTIONS, dtype=np.float64)
        ev_actions[FOLD] = ev_fold
        ev_actions[CALL] = ev_call
        # All raise actions map to same EV as call (depth limit)
        for a in (RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN):
            ev_actions[a] = ev_call

        p_fold = strat_fc[FOLD]
        node_ev = p_fold * ev_fold + (1 - p_fold) * ev_call

        self.regret[2] = np.maximum(0.0, self.regret[2] + ev_actions - node_ev)

        return node_ev

    def run(self, n_iterations: int) -> None:
        for t in range(n_iterations):
            self._node0(
                t,
                sb=STARTING_STACK - SB_POST,
                bb=STARTING_STACK - BB_POST,
                pot=SB_POST + BB_POST,
            )

    def avg_strategy(self) -> np.ndarray:
        """Return (2, 5) float32 average strategy: [SB node, BB response node]."""
        result = np.zeros((2, N_ACTIONS), dtype=np.float32)
        for pos, node in enumerate([0, 1]):
            s = self.strat_sum[node]
            total = s.sum()
            if total > 0:
                result[pos] = (s / total).astype(np.float32)
            else:
                result[pos] = np.float32(1.0 / N_ACTIONS)
        return result


def solve_bucket(bucket: int, n_iterations: int = 50_000) -> np.ndarray:
    """Run CFR+ for one bucket; return (2, 5) float32 average strategy."""
    solver = _PreflopCFR(float(BUCKET_EQUITY[bucket]))
    solver.run(n_iterations)
    return solver.avg_strategy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_iterations: int = 50_000) -> None:
    out_path = os.path.join(os.path.dirname(__file__), "tables", "preflop_strategy.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    table = np.zeros((169, 2, N_ACTIONS), dtype=np.float32)

    for bucket in range(169):
        table[bucket] = solve_bucket(bucket, n_iterations)

        if bucket == 0 or (bucket + 1) % 17 == 0 or bucket == 168:
            pct = (bucket + 1) / 169 * 100
            aa_allin  = table[12, 0, RAISE_ALLIN]
            low_allin = table[0,  0, RAISE_ALLIN]
            print(f"  bucket {bucket+1:3d}/169 ({pct:.0f}%)  "
                  f"AA allin={aa_allin:.3f}  22 allin={low_allin:.3f}",
                  flush=True)

    np.save(out_path, table)
    print(f"\nSaved: {out_path}  shape={table.shape}", flush=True)

    aa_allin_prob  = table[12, 0, RAISE_ALLIN]
    low_allin_prob = table[0,  0, RAISE_ALLIN]
    print(f"AA  raise-allin probability (SB): {aa_allin_prob:.4f}")
    print(f"22  raise-allin probability (SB): {low_allin_prob:.4f}")
    assert aa_allin_prob > low_allin_prob, (
        f"Sanity check failed: AA allin ({aa_allin_prob:.4f}) "
        f"<= 22 allin ({low_allin_prob:.4f})"
    )
    print("Sanity check passed: AA raises allin more than 22.", flush=True)


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Starting preflop CFR+ computation (50 000 iterations/bucket, 169 buckets)...",
          flush=True)
    main(n_iterations=50_000)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)
