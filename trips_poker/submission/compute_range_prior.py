"""
Compute non-uniform opponent range prior for the 27-card poker game.

Given our 5 dealt cards and the 3-card flop (8 known cards), compute the
probability distribution over all 351 possible opponent 2-card kept hands.

For each of the C(19,5) = 11628 possible opponent 5-card deals from the
remaining 19 cards, we look up which 2-card keep is optimal given the flop,
then count occurrences. The result is a probability vector over C(27,2) = 351
two-card hand indices.

Performance target: < 50ms per call using vectorized numpy.
"""

import numpy as np
from itertools import combinations
from math import comb

NUM_CARDS = 27

# Precompute C(n, k) for n=0..27, k=0..7
_COMB = [[0] * 8 for _ in range(28)]
for _n in range(28):
    _COMB[_n][0] = 1
    for _k in range(1, min(_n + 1, 8)):
        _COMB[_n][_k] = _COMB[_n - 1][_k - 1] + _COMB[_n - 1][_k]

_COMB_NP = np.array(_COMB, dtype=np.int64)  # (28, 8)

NUM_RANKS = 9   # 2..9, A
NUM_SUITS = 3   # d, h, s

# The 10 keep-pair indices within a sorted 5-card hand: C(5,2) combos
_KEEP_PAIRS = list(combinations(range(5), 2))  # [(0,1),(0,2),...,(3,4)]
_KEEP_PAIRS_NP = np.array(_KEEP_PAIRS, dtype=np.int32)  # (10, 2)

# Straight bitmasks: each set of 5 consecutive ranks that form a valid straight
# Ranks: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=A
# Ace wraps low: A-2-3-4-5 (ranks 8,0,1,2,3)
# Ace wraps high: 6-7-8-9-A (ranks 4,5,6,7,8)
_STRAIGHT_MASKS = np.array([
    (1 << 8) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),  # A-2-3-4-5
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4),  # 2-3-4-5-6
    (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5),  # 3-4-5-6-7
    (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6),  # 4-5-6-7-8
    (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7),  # 5-6-7-8-9
    (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8),  # 6-7-8-9-A
], dtype=np.int32)

# High rank of each straight pattern (for tiebreaking)
_STRAIGHT_HIGH = np.array([3, 4, 5, 6, 7, 8], dtype=np.int32)


def _score_keeps_batch(deals: np.ndarray, flop: np.ndarray) -> np.ndarray:
    """Score all 10 keeps for each deal using actual flop-aware hand evaluation.

    For each of the 10 possible 2-card keeps from a 5-card deal, evaluates the
    5-card hand (2 kept + 3 flop) for made hand strength + draw potential.

    Args:
        deals: (N, 5) int32, sorted 5-card hands
        flop: (3,) int32, sorted flop cards

    Returns:
        (N, 10) float32 scores. Higher = better keep.
    """
    N = deals.shape[0]

    # Extract kept cards for all 10 keep positions: (N, 10)
    keeps_c0 = deals[:, _KEEP_PAIRS_NP[:, 0]]  # (N, 10)
    keeps_c1 = deals[:, _KEEP_PAIRS_NP[:, 1]]  # (N, 10)

    # Build 5-card eval hands: (N, 10, 5) = [kept0, kept1, flop0, flop1, flop2]
    flop_exp = np.broadcast_to(flop[np.newaxis, np.newaxis, :], (N, 10, 3))
    eval_hands = np.concatenate([
        keeps_c0[:, :, np.newaxis],
        keeps_c1[:, :, np.newaxis],
        flop_exp
    ], axis=2)  # (N, 10, 5)

    # Ranks and suits
    ranks = eval_hands % NUM_RANKS   # (N, 10, 5)
    suits = eval_hands // NUM_RANKS  # (N, 10, 5)

    # ── Rank counting: (N, 10, 9) ──
    rank_counts = np.zeros((N, 10, NUM_RANKS), dtype=np.int32)
    for r in range(NUM_RANKS):
        rank_counts[:, :, r] = (ranks == r).sum(axis=2)

    max_count = rank_counts.max(axis=2)  # (N, 10)
    n_pairs = (rank_counts == 2).sum(axis=2)  # (N, 10)

    # ── Suit counting: (N, 10, 3) ──
    suit_counts = np.zeros((N, 10, NUM_SUITS), dtype=np.int32)
    for s in range(NUM_SUITS):
        suit_counts[:, :, s] = (suits == s).sum(axis=2)

    max_suit = suit_counts.max(axis=2)  # (N, 10)

    # ── Straight detection via bitmask ──
    rank_bits = np.zeros((N, 10), dtype=np.int32)
    for c in range(5):
        rank_bits |= (1 << ranks[:, :, c])

    is_straight = np.zeros((N, 10), dtype=bool)
    straight_high = np.zeros((N, 10), dtype=np.int32)
    for si in range(len(_STRAIGHT_MASKS)):
        mask = int(_STRAIGHT_MASKS[si])
        match = (rank_bits & mask) == mask
        is_straight |= match
        straight_high = np.where(match, _STRAIGHT_HIGH[si], straight_high)

    # ── Made hand categories ──
    is_flush = (max_suit >= 5)
    has_trips = (max_count >= 3)
    has_full_house = has_trips & (n_pairs >= 1)  # trips + pair
    has_two_pair = (n_pairs >= 2) & ~has_trips
    has_pair = (n_pairs >= 1) & ~has_trips & ~has_two_pair

    is_straight_flush = is_straight & is_flush

    # Best rank for tiebreaking: highest rank with max count
    # For trips: the trip rank; for pairs: the pair rank; etc.
    best_rank = np.zeros((N, 10), dtype=np.float32)
    for r in range(NUM_RANKS):
        best_rank = np.where(rank_counts[:, :, r] >= max_count, r, best_rank)

    # Pair rank for full house tiebreaking
    pair_rank = np.zeros((N, 10), dtype=np.float32)
    for r in range(NUM_RANKS):
        pair_rank = np.where(rank_counts[:, :, r] == 2, r, pair_rank)

    # High card kicker (max rank in hand)
    high_card = ranks.max(axis=2).astype(np.float32)  # (N, 10)

    # ── Made hand score ──
    # Ranking: SF > FH > Flush > Straight > Trips > TwoPair > Pair > HC
    # (matches WrappedEval score ranges in card_utils.py)
    score = np.full((N, 10), 1000.0, dtype=np.float32)  # default: high card
    score = np.where(has_pair, 3000.0, score)
    score = np.where(has_two_pair, 4000.0, score)
    score = np.where(has_trips, 5000.0, score)
    score = np.where(is_straight, 6000.0, score)
    score = np.where(is_flush, 7000.0, score)
    score = np.where(has_full_house, 8000.0, score)  # overrides trips
    score = np.where(is_straight_flush, 9000.0, score)

    # Add tiebreakers: best rank * 10 + pair rank (for FH) or high card
    score += best_rank * 10.0
    score += np.where(has_full_house, pair_rank, high_card)

    # ── Draw bonuses (turn + river to come) ──
    # Flush draw: 4 suited → ~35% to hit flush (strong draw)
    flush_draw = (max_suit == 4) & ~is_flush
    score += np.where(flush_draw, 500.0, 0.0)

    # Flush backdoor: 3 suited → ~4% to hit flush (minor draw)
    flush_backdoor = (max_suit == 3) & ~is_flush & ~flush_draw
    score += np.where(flush_backdoor, 30.0, 0.0)

    # Straight draw: 4/5 bits match any straight pattern
    has_oesd = np.zeros((N, 10), dtype=bool)
    for si in range(len(_STRAIGHT_MASKS)):
        mask = int(_STRAIGHT_MASKS[si])
        # Count how many of the 5 ranks hit this pattern
        match_count = np.zeros((N, 10), dtype=np.int32)
        for b in range(NUM_RANKS):
            if mask & (1 << b):
                match_count += ((rank_bits >> b) & 1)
        has_oesd |= (match_count >= 4) & ~is_straight

    score += np.where(has_oesd, 300.0, 0.0)

    # Combo draw bonus: flush draw + straight draw together is very strong
    combo_draw = flush_draw & has_oesd
    score += np.where(combo_draw, 200.0, 0.0)

    return score


def _compute_optimal_keeps_realtime(deals: np.ndarray, flop: np.ndarray) -> np.ndarray:
    """Compute optimal keep index (0-9) for each deal using real-time flop evaluation.

    Replaces OPTIMAL_KEEP_TABLE lookup with actual hand + draw evaluation on the
    known flop. Much more accurate than hs2 heuristic (~47% agreement) because
    it evaluates made hands and draws using actual flop cards.

    Args:
        deals: (N, 5) int32, sorted 5-card hands
        flop: (3,) int32, sorted flop cards

    Returns:
        (N,) int32 array of optimal keep indices (0-9).
    """
    scores = _score_keeps_batch(deals, flop)  # (N, 10)
    return np.argmax(scores, axis=1).astype(np.int32)


def _combo_index_2_scalar(c0: int, c1: int) -> int:
    """combo_index for a sorted 2-card hand."""
    # s = sorted pair, index = C(s[0],1) + C(s[1],2)
    if c0 > c1:
        c0, c1 = c1, c0
    return _COMB[c0][1] + _COMB[c1][2]


def _combo_index_5_batch(cards: np.ndarray) -> np.ndarray:
    """Vectorized combo_index for (N, 5) sorted arrays. Returns (N,) int64."""
    idx = np.zeros(cards.shape[0], dtype=np.int64)
    for i in range(5):
        idx += _COMB_NP[cards[:, i], i + 1]
    return idx


def _combo_index_2_batch(cards: np.ndarray) -> np.ndarray:
    """Vectorized combo_index for (N, 2) sorted arrays. Returns (N,) int64."""
    return _COMB_NP[cards[:, 0], 1] + _COMB_NP[cards[:, 1], 2]


def _local_flop_index_batch(hands_5: np.ndarray, flop_sorted: np.ndarray) -> np.ndarray:
    """Vectorized local flop index for N hands.

    For each 5-card hand, compute the position of each flop card within the
    22 remaining cards (cards in 0..26 not in that hand), then compute the
    combinatorial index C(p0,1) + C(p1,2) + C(p2,3).

    Args:
        hands_5: (N, 5) sorted int32 array of 5-card hands
        flop_sorted: (3,) sorted int32 array of flop cards

    Returns:
        (N,) int64 array of local flop indices
    """
    N = hands_5.shape[0]
    # For each flop card, find its position among the 22 remaining cards.
    # Position of card c in remaining = c - (number of hand cards < c).
    # Since hands_5 is sorted, we can use searchsorted.
    positions = np.empty((N, 3), dtype=np.int64)
    for fi in range(3):
        c = int(flop_sorted[fi])
        # Position of card c among the 22 remaining cards (0..26 minus hand_5)
        # equals c minus the number of hand cards with value < c.
        count_below = np.zeros(N, dtype=np.int64)
        for col in range(5):
            count_below += (hands_5[:, col] < c).astype(np.int64)
        positions[:, fi] = c - count_below

    # Sort positions along axis=1 (should already be sorted if flop is sorted
    # and hands don't interleave with flop, but be safe)
    positions.sort(axis=1)

    # Compute combinatorial index: C(p0,1) + C(p1,2) + C(p2,3)
    return (_COMB_NP[positions[:, 0], 1] +
            _COMB_NP[positions[:, 1], 2] +
            _COMB_NP[positions[:, 2], 3])


def _get_deals_and_flop(our_5_cards, flop_3):
    """Common setup: enumerate opponent deals and prepare flop array."""
    known = set(our_5_cards) | set(flop_3)
    assert len(known) == 8, f"Expected 8 unique known cards, got {len(known)}"
    remaining = np.array([c for c in range(NUM_CARDS) if c not in known], dtype=np.int32)
    assert len(remaining) == 19
    flop_sorted = np.array(sorted(flop_3), dtype=np.int32)
    deals_list = list(combinations(remaining.tolist(), 5))
    deals = np.array(deals_list, dtype=np.int32)  # (11628, 5), already sorted
    return deals, flop_sorted


def _optimal_keeps_from_deals(deals, flop_sorted):
    """Compute optimal keep positions using real-time flop evaluation.

    Returns (ki, kj) arrays: index into each deal's 5-card hand for the kept pair.
    """
    optimal_keep_pos = _compute_optimal_keeps_realtime(deals, flop_sorted)
    ki = _KEEP_PAIRS_NP[optimal_keep_pos, 0]
    kj = _KEEP_PAIRS_NP[optimal_keep_pos, 1]
    return ki, kj


def compute_opponent_prior_weighted(our_5_cards, flop_3, hand_weights):
    """Like compute_opponent_prior but weights each 5-card deal by hand_weights[combo_index_5].

    Uses real-time flop-aware hand evaluation instead of precomputed hs2 table.
    """
    deals, flop_sorted = _get_deals_and_flop(our_5_cards, flop_3)
    ki, kj = _optimal_keeps_from_deals(deals, flop_sorted)

    kept_c0 = deals[np.arange(len(deals)), ki]
    kept_c1 = deals[np.arange(len(deals)), kj]
    kept_hand_indices = _COMB_NP[kept_c0, 1] + _COMB_NP[kept_c1, 2]

    # Weight each 5-card deal by its preflop likelihood
    hand_indices = _combo_index_5_batch(deals)
    weights = hand_weights[hand_indices].astype(np.float64)

    prior = np.zeros(comb(NUM_CARDS, 2), dtype=np.float64)  # 351
    np.add.at(prior, kept_hand_indices, weights)
    total = prior.sum()
    if total > 0:
        prior /= total

    return prior


def compute_opponent_prior(our_5_cards, flop_3):
    """Compute the prior probability of each opponent 2-card kept hand.

    Given the 8 known cards (our 5 dealt cards + 3 flop cards), enumerate all
    C(19,5) = 11628 possible opponent 5-card deals from the remaining 19 cards.
    For each deal, compute the optimal 2-card keep using real-time flop-aware
    hand evaluation (made hand + draws). Count how often each 2-card hand index
    is the optimal keep, and normalize.

    Much more accurate than the precomputed hs2 table (~47% agreement with
    equity-based optimal_discard). This evaluates actual made hands and draw
    potential on the known flop.

    Performance: ~200-500ms per call (vectorized numpy).
    """
    deals, flop_sorted = _get_deals_and_flop(our_5_cards, flop_3)
    ki, kj = _optimal_keeps_from_deals(deals, flop_sorted)

    kept_c0 = deals[np.arange(len(deals)), ki]
    kept_c1 = deals[np.arange(len(deals)), kj]
    kept_hand_indices = _COMB_NP[kept_c0, 1] + _COMB_NP[kept_c1, 2]

    prior = np.zeros(comb(NUM_CARDS, 2), dtype=np.float64)  # 351
    np.add.at(prior, kept_hand_indices, 1.0)
    total = prior.sum()
    if total > 0:
        prior /= total

    return prior


def test():
    """Verify the prior sums to ~1.0 and check performance."""
    import time

    our_5 = [0, 1, 2, 3, 4]
    flop_3 = [9, 10, 11]

    # Warm up
    prior = compute_opponent_prior(our_5, flop_3)

    # Time it
    N_TRIALS = 10
    start = time.perf_counter()
    for _ in range(N_TRIALS):
        prior = compute_opponent_prior(our_5, flop_3)
    elapsed = (time.perf_counter() - start) / N_TRIALS * 1000

    print(f"Prior shape: {prior.shape}")
    print(f"Prior sum: {prior.sum():.10f}")
    print(f"Non-zero entries: {np.count_nonzero(prior)} / {len(prior)}")
    print(f"Min non-zero prob: {prior[prior > 0].min():.6f}")
    print(f"Max prob: {prior.max():.6f}")
    print(f"Time per call: {elapsed:.2f} ms")

    assert prior.shape == (351,), f"Bad shape: {prior.shape}"
    assert abs(prior.sum() - 1.0) < 1e-10, f"Prior doesn't sum to 1: {prior.sum()}"
    assert np.all(prior >= 0), "Negative probabilities"

    # Hands that include any of our known cards should have zero probability
    known = set(our_5) | set(flop_3)
    for c0 in range(NUM_CARDS):
        for c1 in range(c0 + 1, NUM_CARDS):
            idx = _COMB[c0][1] + _COMB[c1][2]
            if c0 in known or c1 in known:
                assert prior[idx] == 0.0, (
                    f"Hand ({c0},{c1}) includes known card but has prob {prior[idx]}")

    # Test with different cards
    our_5b = [18, 19, 20, 21, 22]
    flop_3b = [0, 1, 2]
    prior_b = compute_opponent_prior(our_5b, flop_3b)
    assert abs(prior_b.sum() - 1.0) < 1e-10, f"Prior B doesn't sum to 1: {prior_b.sum()}"
    print(f"\nTest 2 non-zero entries: {np.count_nonzero(prior_b)} / {len(prior_b)}")

    # The distribution should NOT be uniform
    nonzero_probs = prior[prior > 0]
    assert nonzero_probs.max() / nonzero_probs.min() > 1.0, (
        "Distribution is perfectly uniform")
    print(f"Max/min ratio: {nonzero_probs.max() / nonzero_probs.min():.2f}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test()
