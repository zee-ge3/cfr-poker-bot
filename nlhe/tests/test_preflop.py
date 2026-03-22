import numpy as np
import pytest
from nlhe.cfr.preflop import PreflopTable


def test_table_loads():
    table = PreflopTable()
    assert table.strategy.shape == (169, 2, 5)


def test_aa_raises_more_than_72o():
    table = PreflopTable()
    from nlhe.cfr.abstraction import RAISE_ALLIN, FOLD
    # AA bucket = 12 (highest pair), 72o is a low offsuit hand
    aa_raise = table.strategy[12, 0, RAISE_ALLIN]
    # 72o: rank 5 (7) and rank 0 (2), offsuit -> find its bucket
    from nlhe.cfr.abstraction import HAND_BUCKET, ALL_HANDS, HAND_TO_IDX
    # Find a 7-high offsuit hand (e.g. 7c 2d -> indices)
    sevs = [c for c in range(52) if c // 4 == 5]  # 7s have rank 5 (0=2,1=3,...,5=7)
    twos = [c for c in range(52) if c // 4 == 0]
    h72 = (min(sevs[0], twos[0]), max(sevs[0], twos[0]))
    bucket_72 = HAND_BUCKET[HAND_TO_IDX[h72]]
    low_raise = table.strategy[bucket_72, 0, RAISE_ALLIN]
    assert aa_raise > low_raise, f"AA ({aa_raise:.3f}) should raise more than 72o ({low_raise:.3f})"


def test_lookup_returns_valid_prob():
    table = PreflopTable()
    probs = table.lookup(our_hole_idxs=(48, 49), position=0)  # AS AH
    assert abs(probs.sum() - 1.0) < 1e-5
    assert all(p >= 0 for p in probs)
