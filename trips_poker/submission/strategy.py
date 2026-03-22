"""
Adaptive strategy engine V7.
Optimized for 27-card Trips Poker with Blocker-Aware Preflop and RHS.
"""
import math
import random
from collections import Counter
from submission.card_utils import suit, rank, NUM_RANKS

def soft_decision(value: float, threshold: float, temp: float = 0.04) -> float:
    x = (value - threshold) / temp
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _ace_wrap_connected(r1: int, r2: int) -> bool:
    diff = abs(r1 - r2)
    if diff <= 2: return True
    if r1 == 8 or r2 == 8:
        other = r2 if r1 == 8 else r1
        return other <= 1 or other >= 6
    return False

def _board_wetness(board: list) -> float:
    suits_list = [suit(c) for c in board if c >= 0]
    ranks_list = sorted([rank(c) for c in board if c >= 0])
    if not suits_list: return 0.5
    flush_score = max(Counter(suits_list).values()) / len(suits_list)
    connect_score = sum(1 for i in range(len(ranks_list)-1) if _ace_wrap_connected(ranks_list[i], ranks_list[i+1])) / max(1, len(ranks_list)-1)
    return (flush_score + connect_score) / 2

def decide(equity: float, pot_odds: float, street: int, pot: int,
           cost: int, min_raise: int, max_raise: int,
           valid_actions: list, opp_ctx: dict, match_state: dict,
           board: list, persona: dict, facing_raise: bool,
           in_position: bool = True) -> tuple:
    """V9.4: Only called for preflop. Postflop is pure CFR in player.py."""

    if match_state.get('in_lockout', False):
        return (2, 0, 0, 0) if valid_actions[2] else (0, 0, 0, 0)

    return _decide_preflop(equity, pot_odds, valid_actions, cost, min_raise, max_raise, pot, opp_ctx, match_state, persona, in_position)

def _decide_preflop(equity, pot_odds, valid_actions, cost, min_raise, max_raise, pot, opp_ctx, match_state, persona, in_position):
    # V9: Mixed strategy preflop via soft_decision (sigmoid).
    # Old bug: call_thr=0.35 (never folds, preflop eq always >0.43),
    # three_bet_thr=0.78 (never raises). Fixed thresholds, kept mixing.

    if cost > 0:
        # SB blind (cost=1) or facing actual raise (cost>1).
        # Tighter thresholds against bigger raises.
        raise_cost_adj = min(0.06, max(0, cost - 1) * 0.03)
        fold_thr = (0.47 if in_position else 0.50) + raise_cost_adj
        raise_thr = (0.56 if in_position else 0.59) + raise_cost_adj

        # Raise strong — mixed around threshold
        if valid_actions[1] and max_raise >= min_raise:
            if random.random() < soft_decision(equity, raise_thr):
                amt = max(min_raise, min(max_raise, int(pot * 1.0)))
                return (1, amt, 0, 0)

        # Call or fold — mixed around threshold
        if random.random() < soft_decision(equity, fold_thr):
            return (3, 0, 0, 0) if valid_actions[3] else (0, 0, 0, 0)
        return (0, 0, 0, 0) if valid_actions[0] else (3, 0, 0, 0)

    else:
        # BB (cost=0): check is free. Raise strong hands for value.
        raise_thr = 0.55 if in_position else 0.58
        if valid_actions[1] and max_raise >= min_raise:
            if random.random() < soft_decision(equity, raise_thr):
                amt = max(min_raise, min(max_raise, int(pot * 0.75)))
                return (1, amt, 0, 0)

        return (2, 0, 0, 0) if valid_actions[2] else (3, 0, 0, 0)
