"""
Equity calculator for NLHE hands using the treys library.

Card index convention (matches abstraction.py):
  card_idx = rank * 4 + suit
  rank: 0=2, 1=3, ..., 12=A
  suit: 0=c, 1=d, 2=h, 3=s

Public API
----------
card_str_to_idx   : treys card string ('Ah') -> 0-51 index
idx_to_treys      : 0-51 index -> treys card integer
equity_river_exact: exact equity enumeration on a 5-card board
equity_mc         : Monte Carlo equity (0-4 board cards)
equity_vs_range   : equity vs a weighted opponent range
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Sequence

import numpy as np
from treys import Card, Evaluator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Ordered rank characters matching rank 0..12
_RANKS = '23456789TJQKA'
_SUITS = 'cdhs'

# Build the full 52-card deck as treys integers, ordered by card_idx (rank*4+suit)
_ALL_TREYS_CARDS: list[int] = [
    Card.new(r + s) for r in _RANKS for s in _SUITS
]

# Module-level evaluator — do not create a new Evaluator per call
_evaluator = Evaluator()

# ---------------------------------------------------------------------------
# Index conversion helpers
# ---------------------------------------------------------------------------

# Pre-build lookup tables for O(1) conversion
_STR_TO_IDX: dict[str, int] = {}
_IDX_TO_STR: list[str] = [''] * 52

for _rank_i, _r in enumerate(_RANKS):
    for _suit_i, _s in enumerate(_SUITS):
        _cs = _r + _s
        _idx = _rank_i * 4 + _suit_i
        _STR_TO_IDX[_cs] = _idx
        _IDX_TO_STR[_idx] = _cs


def card_str_to_idx(card_str: str) -> int:
    """Convert a treys-style card string (e.g. 'Ah', 'Kd') to a 0-51 index.

    Index formula: rank * 4 + suit
      rank 0=2, 1=3, ..., 12=A
      suit 0=c, 1=d, 2=h, 3=s
    """
    return _STR_TO_IDX[card_str]


def idx_to_treys(idx: int) -> int:
    """Convert a 0-51 card index to the treys card integer."""
    return _ALL_TREYS_CARDS[idx]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cards_to_treys(card_strs: Sequence[str]) -> list[int]:
    """Convert a list of card strings to treys integers."""
    return [_ALL_TREYS_CARDS[_STR_TO_IDX[c]] for c in card_strs]


def _dead_set(hole: Sequence[str], board: Sequence[str]) -> set[int]:
    """Return the set of card indices blocked by hole + board cards."""
    return {_STR_TO_IDX[c] for c in hole} | {_STR_TO_IDX[c] for c in board}


# ---------------------------------------------------------------------------
# equity_river_exact
# ---------------------------------------------------------------------------

def equity_river_exact(our_hole: list[str], board: list[str]) -> float:
    """Compute exact equity on a fully dealt (5-card) board.

    Enumerates ALL remaining two-card opponent hands and compares
    treys evaluator scores (lower score = better hand).

    Parameters
    ----------
    our_hole : list of 2 card strings
    board    : list of 5 card strings

    Returns
    -------
    float : (wins + 0.5 * ties) / total_opponents
    """
    if len(board) != 5:
        raise ValueError(f"equity_river_exact requires exactly 5 board cards, got {len(board)}")

    our_treys = _cards_to_treys(our_hole)
    board_treys = _cards_to_treys(board)

    our_score = _evaluator.evaluate(board_treys, our_treys)

    dead = _dead_set(our_hole, board)
    live_idxs = [i for i in range(52) if i not in dead]

    wins = ties = total = 0
    for opp_c1, opp_c2 in combinations(live_idxs, 2):
        opp_treys = [_ALL_TREYS_CARDS[opp_c1], _ALL_TREYS_CARDS[opp_c2]]
        opp_score = _evaluator.evaluate(board_treys, opp_treys)
        total += 1
        if our_score < opp_score:
            wins += 1
        elif our_score == opp_score:
            ties += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


# ---------------------------------------------------------------------------
# equity_mc
# ---------------------------------------------------------------------------

def equity_mc(
    our_hole: list[str],
    board: list[str],
    n_samples: int = 2000,
) -> float:
    """Monte Carlo equity estimate.

    Handles boards with 0-4 cards (runout is sampled randomly).
    For a 5-card board, no runout sampling is needed.

    Parameters
    ----------
    our_hole  : list of 2 card strings
    board     : list of 0-5 card strings (will be completed to 5)
    n_samples : number of Monte Carlo samples

    Returns
    -------
    float : (wins + 0.5 * ties) / n_samples
    """
    our_treys = _cards_to_treys(our_hole)
    board_treys_fixed = _cards_to_treys(board)
    runout_needed = 5 - len(board)

    dead_base = _dead_set(our_hole, board)
    live_all = [i for i in range(52) if i not in dead_base]

    wins = ties = 0
    for _ in range(n_samples):
        # Sample opponent hand + runout from live cards
        n_draw = 2 + runout_needed
        sampled = random.sample(live_all, n_draw)
        opp_idxs = sampled[:2]
        runout_idxs = sampled[2:]

        opp_treys = [_ALL_TREYS_CARDS[i] for i in opp_idxs]
        full_board = board_treys_fixed + [_ALL_TREYS_CARDS[i] for i in runout_idxs]

        our_score = _evaluator.evaluate(full_board, our_treys)
        opp_score = _evaluator.evaluate(full_board, opp_treys)

        if our_score < opp_score:
            wins += 1
        elif our_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / n_samples


# ---------------------------------------------------------------------------
# equity_vs_range
# ---------------------------------------------------------------------------

def equity_vs_range(
    our_hole: list[str],
    board: list[str],
    opp_range: np.ndarray,
    n_samples: int = 2000,
) -> float:
    """Equity vs a weighted opponent range (shape [1326,]).

    Samples opponent hands proportional to range weights, then completes the
    runout via Monte Carlo. Returns expected equity.

    Parameters
    ----------
    our_hole  : list of 2 card strings
    board     : list of 0-5 card strings
    opp_range : np.ndarray of shape (1326,) — probability vector (need not sum to 1;
                will be normalised internally). Hands involving dead cards must
                already be zeroed by the caller or will be masked here.
    n_samples : number of Monte Carlo samples

    Returns
    -------
    float : (wins + 0.5 * ties) / n_samples
    """
    from nlhe.cfr.abstraction import ALL_HANDS

    our_treys = _cards_to_treys(our_hole)
    board_treys_fixed = _cards_to_treys(board)
    runout_needed = 5 - len(board)

    dead_base = _dead_set(our_hole, board)
    dead_base_idxs = set(dead_base)

    # Build masked range: zero out hands that contain dead cards
    masked = opp_range.copy().astype(np.float64)
    for hand_idx, (c1, c2) in enumerate(ALL_HANDS):
        if c1 in dead_base_idxs or c2 in dead_base_idxs:
            masked[hand_idx] = 0.0

    total_weight = masked.sum()
    if total_weight <= 0.0:
        return 0.5

    masked /= total_weight

    # Build cumulative distribution for weighted sampling
    hand_indices = np.arange(1326)
    nonzero_mask = masked > 0.0
    nz_hand_idxs = hand_indices[nonzero_mask]
    nz_weights = masked[nonzero_mask]

    wins = ties = 0
    for _ in range(n_samples):
        # Sample opponent hand from weighted range
        chosen_hand_idx = int(np.random.choice(nz_hand_idxs, p=nz_weights))
        opp_c1, opp_c2 = ALL_HANDS[chosen_hand_idx]
        opp_treys = [_ALL_TREYS_CARDS[opp_c1], _ALL_TREYS_CARDS[opp_c2]]

        # Build live set excluding opp cards for runout
        opp_dead = dead_base_idxs | {opp_c1, opp_c2}
        live_for_runout = [i for i in range(52) if i not in opp_dead]

        if runout_needed > 0:
            runout_idxs = random.sample(live_for_runout, runout_needed)
            full_board = board_treys_fixed + [_ALL_TREYS_CARDS[i] for i in runout_idxs]
        else:
            full_board = board_treys_fixed

        our_score = _evaluator.evaluate(full_board, our_treys)
        opp_score = _evaluator.evaluate(full_board, opp_treys)

        if our_score < opp_score:
            wins += 1
        elif our_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / n_samples
