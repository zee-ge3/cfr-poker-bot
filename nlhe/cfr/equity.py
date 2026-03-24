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
equity_vs_range_batch : per-hand (1326,) equity against weighted opponent range
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

# Precomputed tables for vectorized equity_vs_range_batch (lazy init)
_CARD_IN_HAND: np.ndarray | None = None   # (52, 1326) uint8
_HAND_CONFLICT: np.ndarray | None = None  # (1326, 1326) bool


def _ensure_hand_tables():
    """Build card-membership and hand-conflict matrices (computed once)."""
    global _CARD_IN_HAND, _HAND_CONFLICT
    if _CARD_IN_HAND is not None:
        return
    from nlhe.cfr.abstraction import ALL_HANDS
    arr = np.array(ALL_HANDS, dtype=np.int32)  # (1326, 2)
    c1, c2 = arr[:, 0], arr[:, 1]
    cih = np.zeros((52, 1326), dtype=np.uint8)
    for c in range(52):
        cih[c] = ((c1 == c) | (c2 == c)).astype(np.uint8)
    _CARD_IN_HAND = cih
    _HAND_CONFLICT = (cih.T @ cih) > 0  # (1326, 1326) bool

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


# ---------------------------------------------------------------------------
# equity_vs_range_batch
# ---------------------------------------------------------------------------

def equity_vs_range_batch(
    board: list[str],
    opp_range: np.ndarray,
    n_samples: int = 500,
) -> np.ndarray:
    """Per-hand equity against a weighted opponent range.

    For each of the 1326 possible hero hands, compute equity against
    *opp_range* on the given board. Hands that conflict with the board
    get equity 0.

    River (5-card board): vectorized exact enumeration using precomputed
    hand scores and matrix operations.
    Non-river: Monte Carlo with vectorized opponent masking.

    Parameters
    ----------
    board     : list of 0-5 card strings
    opp_range : (1326,) float32 opponent range weights (need not sum to 1)
    n_samples : MC samples per hero hand (ignored on river)

    Returns
    -------
    (1326,) float32 equity for each hero hand
    """
    from nlhe.cfr.abstraction import ALL_HANDS
    _ensure_hand_tables()

    board_idxs = {_STR_TO_IDX[c] for c in board}
    board_treys = _cards_to_treys(board)
    is_river = len(board) == 5
    runout_needed = 5 - len(board)

    # Board-blocked hands mask
    board_blocked = np.zeros(1326, dtype=bool)
    for c in board_idxs:
        board_blocked |= _CARD_IN_HAND[c].astype(bool)

    # Opponent range with board-blocked hands zeroed
    opp_w = opp_range.astype(np.float64).copy()
    opp_w[board_blocked] = 0.0

    if is_river:
        # --- Fully vectorized river computation ---
        # Evaluate all non-blocked hands once (~990 evals vs ~990k before)
        hand_scores = np.full(1326, 99999, dtype=np.int32)
        for idx in range(1326):
            if not board_blocked[idx]:
                c1, c2 = ALL_HANDS[idx]
                hand_scores[idx] = _evaluator.evaluate(
                    board_treys,
                    [_ALL_TREYS_CARDS[c1], _ALL_TREYS_CARDS[c2]],
                )

        # Effective opp weights: zero where hero-opp hands share cards
        eff_opp = np.where(~_HAND_CONFLICT, opp_w[None, :], 0.0)  # (1326, 1326)

        # Payoff matrix (lower treys score = stronger hand)
        hs = hand_scores
        payoff = np.where(
            hs[:, None] < hs[None, :], 1.0,
            np.where(hs[:, None] == hs[None, :], 0.5, 0.0),
        )

        numerator = (eff_opp * payoff).sum(axis=1)
        denominator = eff_opp.sum(axis=1)

        equities = np.where(
            denominator > 0, numerator / denominator, 0.5,
        ).astype(np.float32)
        equities[board_blocked] = 0.0
        return equities

    # --- Vectorized MC path (flop / turn) ---
    # Sample board completions, then do exact vectorized enumeration for each.
    # Each runout evaluates ALL opponent hands (not just one sampled hand),
    # so n_samples runouts gives accuracy equivalent to n_samples * ~1000 MC.
    n_runouts = max(10, n_samples)
    equities_sum = np.zeros(1326, dtype=np.float64)
    valid_count = np.zeros(1326, dtype=np.float64)

    live_for_runout = [i for i in range(52) if i not in board_idxs]

    for _ in range(n_runouts):
        runout = random.sample(live_for_runout, runout_needed)
        full_board_treys = board_treys + [_ALL_TREYS_CARDS[c] for c in runout]

        # Hands blocked by the complete board (original board + runout)
        full_blocked = board_blocked.copy()
        for c in runout:
            full_blocked |= _CARD_IN_HAND[c].astype(bool)

        # Evaluate all non-blocked hands on this complete board
        hand_scores = np.full(1326, 99999, dtype=np.int32)
        for idx in range(1326):
            if not full_blocked[idx]:
                c1, c2 = ALL_HANDS[idx]
                hand_scores[idx] = _evaluator.evaluate(
                    full_board_treys,
                    [_ALL_TREYS_CARDS[c1], _ALL_TREYS_CARDS[c2]],
                )

        # Opponent range with runout blocking
        opp_for_runout = opp_w.copy()
        opp_for_runout[full_blocked] = 0.0

        # Effective opp weights (zero where hero-opp share cards)
        eff_opp = np.where(~_HAND_CONFLICT, opp_for_runout[None, :], 0.0)

        # Payoff matrix
        hs = hand_scores
        payoff = np.where(
            hs[:, None] < hs[None, :], 1.0,
            np.where(hs[:, None] == hs[None, :], 0.5, 0.0),
        )

        numerator = (eff_opp * payoff).sum(axis=1)
        denominator = eff_opp.sum(axis=1)

        eq_this = np.where(denominator > 0, numerator / denominator, 0.5)

        # Accumulate only for non-blocked hero hands
        valid_hero = ~full_blocked
        equities_sum[valid_hero] += eq_this[valid_hero]
        valid_count[valid_hero] += 1.0

    valid = valid_count > 0
    equities = np.full(1326, 0.5, dtype=np.float32)
    equities[valid] = (equities_sum[valid] / valid_count[valid]).astype(np.float32)
    equities[board_blocked] = 0.0
    return equities

def compute_matchup_matrix(board: list[str], n_samples: int = 20) -> np.ndarray:
    """Compute (1326, 1326) equity matrix for all hero hands vs all opp hands.
    Returns M where M[i, j] is the equity of hero hand i against opp hand j.
    M[i, j] = 0 if hands conflict or conflict with the board.
    """
    import os
    from nlhe.cfr.abstraction import ALL_HANDS, HAND_BUCKET
    _ensure_hand_tables()
    
    if len(board) == 0:
        # Preflop: load precomputed 169x169 exact Monte Carlo matrix and map it to 1326x1326
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "tables", "equity_matrix_169.npy")
        if os.path.exists(path):
            eq_169 = np.load(path)
            # Map 169x169 to 1326x1326 using HAND_BUCKET
            M = eq_169[HAND_BUCKET[:, None], HAND_BUCKET[None, :]]
            M = np.where(~_HAND_CONFLICT, M, 0.0)
            return M.astype(np.float32)

    board_idxs = {_STR_TO_IDX[c] for c in board}
    board_treys = _cards_to_treys(board)
    is_river = len(board) == 5
    runout_needed = 5 - len(board)

    board_blocked = np.zeros(1326, dtype=bool)
    for c in board_idxs:
        board_blocked |= _CARD_IN_HAND[c].astype(bool)

    if is_river:
        hand_scores = np.full(1326, 99999, dtype=np.int32)
        for idx in range(1326):
            if not board_blocked[idx]:
                c1, c2 = ALL_HANDS[idx]
                hand_scores[idx] = _evaluator.evaluate(
                    board_treys,
                    [_ALL_TREYS_CARDS[c1], _ALL_TREYS_CARDS[c2]],
                )

        hs = hand_scores
        payoff = np.where(
            hs[:, None] < hs[None, :], 1.0,
            np.where(hs[:, None] == hs[None, :], 0.5, 0.0),
        )
        M = np.where(~_HAND_CONFLICT, payoff, 0.0)
        M[board_blocked, :] = 0.0
        M[:, board_blocked] = 0.0
        return M.astype(np.float32)

    n_runouts = max(10, n_samples)
    live_for_runout = [i for i in range(52) if i not in board_idxs]
    
    M_sum = np.zeros((1326, 1326), dtype=np.float32)
    Count = np.zeros((1326, 1326), dtype=np.float32)

    for _ in range(n_runouts):
        runout = random.sample(live_for_runout, runout_needed)
        full_board_treys = board_treys + [_ALL_TREYS_CARDS[c] for c in runout]

        full_blocked = board_blocked.copy()
        for c in runout:
            full_blocked |= _CARD_IN_HAND[c].astype(bool)

        hand_scores = np.full(1326, 99999, dtype=np.int32)
        for idx in range(1326):
            if not full_blocked[idx]:
                c1, c2 = ALL_HANDS[idx]
                hand_scores[idx] = _evaluator.evaluate(
                    full_board_treys,
                    [_ALL_TREYS_CARDS[c1], _ALL_TREYS_CARDS[c2]],
                )

        hs = hand_scores
        payoff = np.where(
            hs[:, None] < hs[None, :], 1.0,
            np.where(hs[:, None] == hs[None, :], 0.5, 0.0),
        )
        
        valid_mask = ~_HAND_CONFLICT.copy()
        valid_mask[full_blocked, :] = False
        valid_mask[:, full_blocked] = False
        
        M_sum[valid_mask] += payoff[valid_mask]
        Count[valid_mask] += 1.0

    M = np.where(Count > 0, M_sum / Count, 0.5)
    M[board_blocked, :] = 0.0
    M[:, board_blocked] = 0.0
    M[_HAND_CONFLICT] = 0.0
    return M.astype(np.float32)
