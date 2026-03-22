# submission/equity.py
"""
Exact equity engine using precomputed hand rank tables.
Zero Monte Carlo sampling — all equity is computed via exhaustive enumeration
with O(1) table lookups.

Vectorized with numpy for performance: ~20ms per optimal_discard call.
"""
import os
import random
import numpy as np
from itertools import combinations
from submission.card_utils import (
    combo_index_2, combo_index_5, combo_index_7,
    NUM_CARDS, COMB, rank, suit
)

_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')

def _load_table(name, dtype=None):
    path = os.path.join(_TABLE_DIR, name)
    if os.path.exists(path):
        return np.load(path)
    return None

HAND_RANKS = _load_table('hand_ranks.npy')
HAND_STRENGTH_2 = _load_table('hand_strength_2.npy')
HAND_POTENTIAL_5 = _load_table('hand_potential_5.npy')
OPTIMAL_KEEP_TABLE = _load_table('optimal_keep_table.npy')  # (80730, 1540) uint8

# Precompute C(n,k) table as numpy array for fast local index computation
_COMB_PY = [[0] * 8 for _ in range(28)]
for _n in range(28):
    _COMB_PY[_n][0] = 1
    for _k in range(1, min(_n + 1, 8)):
        _COMB_PY[_n][_k] = _COMB_PY[_n - 1][_k - 1] + _COMB_PY[_n - 1][_k]


def _local_flop_index(hand_5: tuple, flop: tuple) -> int:
    """Compute the combinatorial index of flop within the 22 cards remaining
    after removing hand_5 from the 27-card deck.

    Returns int in [0, C(22,3)) = [0, 1540).
    Used to index into OPTIMAL_KEEP_TABLE[hand_idx, flop_local_idx].
    """
    remaining = [c for c in range(NUM_CARDS) if c not in hand_5]  # 22 sorted ints
    pos_lookup = {c: i for i, c in enumerate(remaining)}
    p0, p1, p2 = sorted(pos_lookup[c] for c in flop)
    return _COMB_PY[p0][1] + _COMB_PY[p1][2] + _COMB_PY[p2][3]

# Precompute numpy COMB table for vectorized combo_index
_COMB_NP = np.array(COMB, dtype=np.int64)  # shape (28, 8)


def hand_rank(my_2: tuple, board_5: tuple) -> int:
    idx = combo_index_7(tuple(sorted(my_2 + board_5)))
    return int(HAND_RANKS[idx])


def _combo_index_7_batch(cards_array: np.ndarray) -> np.ndarray:
    """Vectorized combo_index_7 for an (N, 7) sorted numpy array."""
    # cards_array: (N, 7), each row sorted ascending
    # combo_index = sum(COMB[card[i]][i+1] for i in range(7))
    indices = np.zeros(cards_array.shape[0], dtype=np.int64)
    for i in range(7):
        indices += _COMB_NP[cards_array[:, i], i + 1]
    return indices


def hand_strength_2_lookup(hand: tuple) -> float:
    if HAND_STRENGTH_2 is None:
        return 0.5
    return float(HAND_STRENGTH_2[combo_index_2(tuple(sorted(hand)))])


def preflop_strength(hole_5: tuple) -> float:
    """Pre-flop hand strength accounting for the discard mechanic.

    With hand_potential_5.npy: returns flop-averaged equity of the optimal
    2-card keep — computed offline over all 1540 possible flops.

    Fallback (no table): best hand_strength_2 among all 10 2-card combos
    from the 5-card hand (ignores flop texture — less accurate).
    """
    if HAND_POTENTIAL_5 is not None:
        return float(HAND_POTENTIAL_5[combo_index_5(tuple(sorted(hole_5)))])
    if HAND_STRENGTH_2 is None:
        return 0.5
    best = 0.0
    for i, j in combinations(range(5), 2):
        keep = (hole_5[i], hole_5[j])
        eq = hand_strength_2_lookup(keep)
        if eq > best:
            best = eq
    return best


def action_range_weight(hand_score: int, action: str) -> float:
    """Bayesian range weight update based on opponent's in-hand action.

    Uses precomputed hand score (lower = stronger, range 1–7462) to estimate
    how likely a rational player is to take a given action with that hand.

    Multiply the existing weight for each candidate hand by this factor,
    then renormalize. Applied after each opponent action within a hand.

    Args:
        hand_score: HAND_RANKS result for opponent hand + board (lower = better)
        action: 'raise', 'call', 'check', 'fold'

    Returns:
        Likelihood weight multiplier (unnormalized, > 0).
    """
    # Normalize score to strength ∈ (0,1): 1 = nuts, 0 = worst
    strength = 1.0 - (hand_score - 1) / 7461.0
    strength = max(0.0, min(1.0, strength))

    if action == 'raise':
        # Strong hands raise: weight rises steeply above median
        return max(0.01, strength ** 1.5)
    elif action == 'call':
        # Medium hands call: peak around strength=0.5
        return max(0.01, 4.0 * strength * (1.0 - strength) + 0.1)
    elif action == 'check':
        # Weak or trapping hands check: bimodal but skewed weak
        return max(0.01, 0.8 * (1.0 - strength) + 0.2)
    elif action == 'fold':
        # Weak hands fold — rarely need to weight, hand is over
        return max(0.01, (1.0 - strength) ** 2)
    return 1.0


def make_action_weights_fn(base_weights_fn, opp_hands, board, hand_actions):
    """Return an updated opp_weights_fn incorporating in-hand action history.

    Args:
        base_weights_fn: existing opp_weights_fn (historical model) or None
        opp_hands: list of (c0,c1) candidate opponent hands
        board: list of community cards seen so far (int card indices)
        hand_actions: list of ('raise'|'call'|'check'|'fold', street) tuples
                      — opponent actions in current hand, in order

    Returns:
        new_weights_fn(hand) -> float, incorporating Bayesian range update.
    """
    if not hand_actions or not board or len(board) < 3:
        return base_weights_fn

    n_board = len(board)

    # Compute a Bayesian update multiplier per candidate hand.
    # For river (5 cards): use exact 7-card HAND_RANKS table.
    # For flop/turn (3-4 cards): use HAND_STRENGTH_2 (2-card prior strength) as proxy.
    hand_multipliers = {}

    if n_board >= 5 and HAND_RANKS is not None:
        # River: full board, exact scoring
        board_5 = tuple(board[:5])
        for hand in opp_hands:
            combined = tuple(sorted(hand + board_5))
            if len(combined) != 7:
                hand_multipliers[hand] = 1.0
                continue
            try:
                score = int(HAND_RANKS[combo_index_7(combined)])
            except (IndexError, TypeError):
                hand_multipliers[hand] = 1.0
                continue

            multiplier = 1.0
            for action, _street in hand_actions:
                multiplier *= action_range_weight(score, action)
            hand_multipliers[hand] = multiplier

    elif HAND_STRENGTH_2 is not None:
        # Flop/turn: approximate via 2-card preflop strength as proxy.
        # HS2 in [~0.35, ~0.75]; remap to strength in [0, 1].
        HS2_MIN, HS2_MAX = 0.34, 0.76
        for hand in opp_hands:
            try:
                hs2 = float(HAND_STRENGTH_2[combo_index_2(tuple(sorted(hand)))])
                strength = max(0.0, min(1.0, (hs2 - HS2_MIN) / (HS2_MAX - HS2_MIN)))
                # Convert to treys-style score (lower=better) for action_range_weight
                fake_score = int(1 + (1.0 - strength) * 7461)
            except (IndexError, TypeError):
                hand_multipliers[hand] = 1.0
                continue

            multiplier = 1.0
            for action, _street in hand_actions:
                multiplier *= action_range_weight(fake_score, action)
            hand_multipliers[hand] = multiplier
    else:
        return base_weights_fn

    def updated_weights_fn(hand):
        base = base_weights_fn(hand) if base_weights_fn else 1.0
        return base * hand_multipliers.get(hand, 1.0)

    return updated_weights_fn


def exact_equity(my_2, board, dead_cards, opp_discards=None,
                 my_discards=None, opp_weights_fn=None,
                 hand_actions=None) -> float:
    """Compute exact equity via exhaustive enumeration with O(1) table lookups.

    Args:
        my_2: tuple of 2 int cards (our kept cards)
        board: list/tuple of community cards seen so far (3-5 cards)
        dead_cards: additional dead cards (our discards, opp discards)
        opp_discards: opponent's 3 discarded cards — enables rational filter
        my_discards: our 3 discarded cards (dead)
        opp_weights_fn: callable(hand) -> float, prior range weights
        hand_actions: list of (action_str, street) tuples for in-hand range
                      narrowing — ('raise'/'call'/'check'/'fold', street_int).
                      Applied as Bayesian updates on top of opp_weights_fn.
    """
    all_dead = set(my_2) | set(board) | set(dead_cards or ())
    remaining = [c for c in range(NUM_CARDS) if c not in all_dead]

    # Build effective weights function: combine prior with in-hand action update.
    # Applied on all post-flop streets (river: exact board scoring; flop/turn: HS2 proxy).
    effective_weights_fn = opp_weights_fn
    if hand_actions and len(board) >= 3:
        flop = tuple(board[:3])
        if opp_discards is not None:
            candidate_hands = list(_rational_opponent_hands(
                remaining, opp_discards, flop, my_discards))
        else:
            candidate_hands = list(combinations(remaining, 2))
        effective_weights_fn = make_action_weights_fn(
            opp_weights_fn, candidate_hands, list(board), hand_actions)

    if len(board) == 5:
        eq = _equity_river(my_2, board, remaining, opp_discards,
                           my_discards, effective_weights_fn)
    elif len(board) == 4:
        eq = _equity_partial_board(my_2, board, remaining, 1,
                                   opp_discards, my_discards, effective_weights_fn)
    elif len(board) == 3:
        eq = _equity_partial_board(my_2, board, remaining, 2,
                                   opp_discards, my_discards, effective_weights_fn)
    else:
        eq = 0.5

    # Fallback: if equity resolved to exactly 0.5 (empty candidate set from rational
    # filter), retry with UNFILTERED candidates (all remaining 2-card combos).
    # This gives real board-aware equity instead of a static HS2 preflop value.
    if eq == 0.5 and len(board) >= 3:
        if len(board) == 5:
            eq = _equity_river(my_2, board, remaining, None, None, None)
        elif len(board) == 4:
            eq = _equity_partial_board(my_2, board, remaining, 1, None, None, None)
        elif len(board) == 3:
            eq = _equity_partial_board(my_2, board, remaining, 2, None, None, None)

    return eq


def _equity_river(my_2, board_5, remaining, opp_discards, my_discards, opp_weights_fn):
    board_5_t = tuple(board_5)
    my_score = hand_rank(my_2, board_5_t)

    if opp_discards is not None:
        opp_hands = list(_rational_opponent_hands(remaining, opp_discards,
                                                   board_5_t[:3], my_discards))
    else:
        opp_hands = list(combinations(remaining, 2))

    if not opp_hands:
        return 0.5

    # Vectorized scoring
    opp_arr = np.array(opp_hands, dtype=np.int32)  # (N, 2)
    board_rep = np.tile(np.array(board_5_t, dtype=np.int32), (len(opp_hands), 1))  # (N, 5)
    all_7 = np.concatenate([opp_arr, board_rep], axis=1)  # (N, 7)
    all_7.sort(axis=1)
    opp_scores = HAND_RANKS[_combo_index_7_batch(all_7)]

    if opp_weights_fn is None:
        wins = np.sum(my_score < opp_scores) + 0.5 * np.sum(my_score == opp_scores)
        return float(wins / len(opp_scores))

    # Weighted version
    weights = np.array([opp_weights_fn(h) for h in opp_hands], dtype=np.float64)
    mask = weights > 0
    if not np.any(mask):
        return 0.5
    opp_scores = opp_scores[mask]
    weights = weights[mask]
    wins = np.sum(weights[my_score < opp_scores]) + 0.5 * np.sum(weights[my_score == opp_scores])
    return float(wins / np.sum(weights))


def _equity_partial_board(my_2, board_partial, remaining, cards_needed,
                          opp_discards, my_discards, opp_weights_fn):
    flop = tuple(board_partial[:3])
    if opp_discards is not None:
        rational_hands = list(_rational_opponent_hands(remaining, opp_discards,
                                                        flop, my_discards))
    else:
        rational_hands = None

    board_exts = list(combinations(remaining, cards_needed))
    if not board_exts:
        return 0.5

    # Build candidate opponent hands
    if rational_hands is not None:
        cand_hands = rational_hands
    else:
        cand_hands = list(combinations(remaining, 2))

    if not cand_hands:
        return 0.5

    cand_arr = np.array(cand_hands, dtype=np.int32)  # (H, 2)
    n_cands = len(cand_hands)

    # Get weights once
    if opp_weights_fn is not None:
        weights_all = np.array([opp_weights_fn(h) for h in cand_hands], dtype=np.float64)
    else:
        weights_all = None  # uniform — skip weight array ops

    # Precompute my scores for all board extensions
    board_partial_t = tuple(board_partial)
    my_2_t = tuple(my_2)
    ext_arr = np.array(board_exts, dtype=np.int32)  # (E, cards_needed)
    n_exts = len(board_exts)

    # My 7-card combos: (E, 7)
    my_rep = np.tile(np.array(my_2_t, dtype=np.int32), (n_exts, 1))  # (E, 2)
    bp_rep = np.tile(np.array(board_partial_t, dtype=np.int32), (n_exts, 1))  # (E, len(bp))
    my_all7 = np.concatenate([my_rep, bp_rep, ext_arr], axis=1)  # (E, 7)
    my_all7.sort(axis=1)
    my_scores = HAND_RANKS[_combo_index_7_batch(my_all7)]  # (E,)

    # Build collision mask: (E, H) — True where opp hand does NOT collide with ext
    # For each extension card, check it doesn't appear in either opp card column
    no_collision = np.ones((n_exts, n_cands), dtype=bool)
    for ci in range(cards_needed):
        ext_col = ext_arr[:, ci:ci+1]  # (E, 1)
        no_collision &= (cand_arr[:, 0:1].T != ext_col) & (cand_arr[:, 1:2].T != ext_col)

    # Apply weight mask
    if weights_all is not None:
        w_mask = weights_all > 0  # (H,)
        no_collision &= w_mask[np.newaxis, :]

    # For each extension, compute opp scores in batch
    # Strategy: process all extensions x candidates in one giant batch
    # Build full_boards for all (ext, cand) pairs where no_collision is True
    ext_indices, cand_indices = np.where(no_collision)
    n_pairs = len(ext_indices)

    if n_pairs == 0:
        return 0.5

    # Build (n_pairs, 7) array
    opp_cards = cand_arr[cand_indices]  # (n_pairs, 2)
    board_cards = np.concatenate([bp_rep[ext_indices], ext_arr[ext_indices]], axis=1)  # (n_pairs, 5)
    all_7 = np.concatenate([opp_cards, board_cards], axis=1)  # (n_pairs, 7)
    all_7.sort(axis=1)
    opp_scores = HAND_RANKS[_combo_index_7_batch(all_7)]  # (n_pairs,)

    # Compare with my scores
    my_scores_expanded = my_scores[ext_indices]  # (n_pairs,)

    if weights_all is not None:
        w = weights_all[cand_indices]
        total_wins = float(np.sum(w[my_scores_expanded < opp_scores]) +
                          0.5 * np.sum(w[my_scores_expanded == opp_scores]))
        total_weight = float(np.sum(w))
    else:
        total_wins = float(np.sum(my_scores_expanded < opp_scores) +
                          0.5 * np.sum(my_scores_expanded == opp_scores))
        total_weight = float(n_pairs)

    return total_wins / total_weight if total_weight > 0 else 0.5


def _flop_aware_keep_score(keep_2, flop_3):
    """Fast flop-aware proxy: hand_strength_2 base + board texture bonuses.
    Captures flush compatibility, board pairs, and straight connectivity
    without expensive runout enumeration."""
    base = hand_strength_2_lookup(keep_2)

    # Board suit distribution
    flop_suits = [suit(c) for c in flop_3]
    keep_suits = [suit(c) for c in keep_2]

    # Flush bonus: on suited boards, matching suit is hugely valuable
    from collections import Counter
    suit_counts = Counter(flop_suits)
    dominant_suit, dominant_count = suit_counts.most_common(1)[0]

    if dominant_count >= 2:  # 2+ of same suit on flop
        matching = sum(1 for s in keep_suits if s == dominant_suit)
        if matching == 2:
            base += 0.30  # two suited = near-certain flush draw
        elif matching == 1:
            base += 0.10  # one suited = backdoor flush draw

    if dominant_count == 3:  # monotone flop
        matching = sum(1 for s in keep_suits if s == dominant_suit)
        if matching == 2:
            base += 0.15  # extra bonus: flush MADE on monotone
        elif matching == 0:
            base -= 0.10  # penalty: no flush possible on monotone

    # Pair with board bonus
    board_ranks = [rank(c) for c in flop_3]
    keep_ranks = [rank(c) for c in keep_2]
    for kr in keep_ranks:
        if kr in board_ranks:
            base += 0.08  # paired with board
            if kr == 8:  # Ace
                base += 0.04  # top pair extra

    # Straight connectivity with board
    all_ranks = sorted(set(board_ranks + keep_ranks))
    if len(all_ranks) >= 4:
        # Check for 4+ in a run (accounting for ace wraps)
        for start_idx in range(len(all_ranks) - 3):
            window = all_ranks[start_idx:start_idx + 4]
            if window[-1] - window[0] <= 4:
                base += 0.06

    return base


def _rational_opponent_hands(remaining, opp_discards, flop, my_discards=None):
    """Yield opponent 2-card hands that a rational player would keep after seeing flop.

    With OPTIMAL_KEEP_TABLE: uses exact precomputed optimal keep (replaces heuristic).
    Fallback: _flop_aware_keep_score heuristic (less accurate).

    A hand (k0, k1) passes if, among all 10 possible 2-card keeps from the
    reconstructed 5-card original hand, (k0, k1) is the optimal keep given the flop.
    """
    opp_disc_list = list(opp_discards)
    flop_t = tuple(sorted(flop))
    use_table = OPTIMAL_KEEP_TABLE is not None

    # Precompute COMB indices for keeps: the 10 pairs of indices into sorted original_5
    _keep_pairs = list(combinations(range(5), 2))  # [(0,1),(0,2),...,(3,4)]

    for hand in combinations(remaining, 2):
        original_5 = tuple(sorted(list(hand) + opp_disc_list))
        if len(original_5) != 5:
            continue

        if use_table:
            # Exact table lookup: which 2-card keep is optimal for this (hand, flop)?
            h_idx = combo_index_5(original_5)
            try:
                local_idx = _local_flop_index(original_5, flop_t)
                optimal_keep_pos = int(OPTIMAL_KEEP_TABLE[h_idx, local_idx])
                # Decode: which 2 cards of original_5 are kept?
                ki, kj = _keep_pairs[optimal_keep_pos]
                optimal_keep = frozenset({original_5[ki], original_5[kj]})
                if frozenset(hand) == optimal_keep:
                    yield hand
            except (KeyError, IndexError, ValueError):
                # Fallback to heuristic if lookup fails
                yield from _rational_hand_heuristic(hand, original_5, flop_t)
        else:
            yield from _rational_hand_heuristic(hand, original_5, flop_t)


def _rational_hand_heuristic(hand, original_5, flop_t):
    """Heuristic rational filter: keeps hand if _flop_aware_keep_score says it's optimal."""
    best_score = -float('inf')
    best_keep_set = None
    for i, j in combinations(range(5), 2):
        keep = (original_5[i], original_5[j])
        score = _flop_aware_keep_score(keep, flop_t)
        if score > best_score:
            best_score = score
            best_keep_set = frozenset(keep)
    if best_keep_set == frozenset(hand):
        yield hand


def optimal_discard(hole_5, flop_3, opp_weights_fn=None):
    keeps = []
    for i, j in combinations(range(5), 2):
        keep = (hole_5[i], hole_5[j])
        discard = [hole_5[k] for k in range(5) if k not in (i, j)]
        dead = tuple(discard)
        eq = exact_equity(keep, flop_3, dead, opp_weights_fn=opp_weights_fn)
        keeps.append((eq, i, j))

    keeps.sort(key=lambda x: x[0], reverse=True)
    best_eq, bi, bj = keeps[0]
    second_eq, si, sj = keeps[1]

    return (bi, bj)
