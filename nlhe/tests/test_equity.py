"""Tests for nlhe.cfr.equity module."""
import numpy as np
import pytest
from nlhe.cfr.equity import (
    card_str_to_idx,
    idx_to_treys,
    equity_river_exact,
    equity_mc,
    equity_vs_range,
)


def test_card_str_to_idx_round_trip():
    """All card indices must be in [0, 52) and unique."""
    seen = set()
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    for r in ranks:
        for s in suits:
            idx = card_str_to_idx(r + s)
            assert 0 <= idx < 52, f"{r+s} -> {idx} out of range"
            assert idx not in seen, f"duplicate idx {idx} for {r+s}"
            seen.add(idx)
    assert len(seen) == 52


def test_card_str_to_idx_known_values():
    """Spot-check specific cards against the rank*4+suit formula."""
    # rank 0=2, suit 0=c → idx 0
    assert card_str_to_idx('2c') == 0
    # rank 0=2, suit 3=s → idx 3
    assert card_str_to_idx('2s') == 3
    # rank 12=A, suit 0=c → idx 48
    assert card_str_to_idx('Ac') == 48
    # rank 12=A, suit 3=s → idx 51
    assert card_str_to_idx('As') == 51
    # rank 11=K, suit 1=d → idx 45
    assert card_str_to_idx('Kd') == 45


def test_idx_to_treys_round_trip():
    """idx_to_treys should produce the same integer as Card.new(card_str)."""
    from treys import Card
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    for r in ranks:
        for s in suits:
            card_str = r + s
            idx = card_str_to_idx(card_str)
            assert idx_to_treys(idx) == Card.new(card_str), (
                f"Round-trip failed for {card_str}"
            )


def test_river_equity_pair_vs_nothing():
    """AA on a K5272Q board (rainbow, no flush/straight) should beat most hands.

    Board: Ks 5d 2h 7c Qd (5 cards — river)
    Our hole: Ah Ac
    """
    our_hole = ['Ah', 'Ac']
    board = ['Ks', '5d', '2h', '7c', 'Qd']
    eq = equity_river_exact(our_hole, board)
    assert eq > 0.5, f"Expected equity > 0.5 for AA vs K5272Q board, got {eq:.4f}"


def test_river_equity_nuts():
    """A straight flush (the nuts) should have equity near 1.

    Board: Kh Qh Jh Th 2c (river)
    Our hole: Ah 9h
    We have a royal flush (A-K-Q-J-T all hearts), which is the best possible hand.
    No opponent can beat or tie this hand on this board.
    """
    our_hole = ['Ah', '9h']
    board = ['Kh', 'Qh', 'Jh', 'Th', '2c']
    eq = equity_river_exact(our_hole, board)
    # Near-nut hand (straight flush to A beats everyone without an equal SF)
    assert eq > 0.85, f"Expected equity > 0.85 for near-nut SF, got {eq:.4f}"


def test_mc_equity_close_to_exact_on_river():
    """MC equity on a river board should be within 0.05 of exact equity."""
    our_hole = ['Ah', 'Ac']
    board = ['Ks', '5d', '2h', '7c', 'Qd']
    exact = equity_river_exact(our_hole, board)
    # Use many samples to reduce variance
    mc = equity_mc(our_hole, board, n_samples=5000)
    assert abs(mc - exact) < 0.05, (
        f"MC ({mc:.4f}) too far from exact ({exact:.4f}), diff={abs(mc-exact):.4f}"
    )


def test_mc_equity_flop():
    """AA on a K52 flop should have equity strictly between 0.5 and 1.0."""
    our_hole = ['Ah', 'Ac']
    board = ['Ks', '5d', '2h']
    eq = equity_mc(our_hole, board, n_samples=3000)
    assert 0.5 < eq < 1.0, (
        f"Expected 0.5 < equity < 1.0 for AA on K52 flop, got {eq:.4f}"
    )


def test_mc_equity_preflop():
    """AA preflop should have equity > 0.7."""
    our_hole = ['Ah', 'Ac']
    board = []
    eq = equity_mc(our_hole, board, n_samples=3000)
    assert eq > 0.7, f"Expected equity > 0.7 for AA preflop, got {eq:.4f}"


def test_uniform_range_equity_sum():
    """equity_vs_range is consistent with exact river equity.

    On a river board, equity_vs_range with a uniform opponent range
    should match equity_river_exact (up to Monte Carlo variance).
    """
    our_hole = ['Ah', 'Ac']
    board = ['Ks', '5d', '2h', '7c', 'Qd']

    # Build uniform range (equal weight for all hands)
    opp_range = np.ones(1326, dtype=np.float32)
    # Zero out hands that conflict with our cards or the board
    from nlhe.cfr.abstraction import ALL_HANDS
    from nlhe.cfr.equity import card_str_to_idx
    dead_idxs = {card_str_to_idx(c) for c in our_hole + board}
    for i, (c1, c2) in enumerate(ALL_HANDS):
        if c1 in dead_idxs or c2 in dead_idxs:
            opp_range[i] = 0.0
    opp_range /= opp_range.sum()

    # Compute equity via range and via exact enumeration
    our_eq = equity_vs_range(our_hole, board, opp_range, n_samples=4000)
    exact_eq = equity_river_exact(our_hole, board)

    # On a river board with enough samples, these should match closely
    assert 0.0 <= our_eq <= 1.0, f"equity_vs_range returned {our_eq} outside [0,1]"
    assert abs(our_eq - exact_eq) < 0.06, (
        f"equity_vs_range ({our_eq:.4f}) should match equity_river_exact ({exact_eq:.4f})"
    )
