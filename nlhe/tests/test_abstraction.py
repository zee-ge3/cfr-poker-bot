import numpy as np
from nlhe.cfr.abstraction import (
    ALL_HANDS, HAND_TO_IDX, HAND_CONTAINS_CARD, HAND_BUCKET
)


def test_hand_count():
    assert len(ALL_HANDS) == 1326


def test_hand_to_idx_round_trip():
    for i, h in enumerate(ALL_HANDS):
        assert HAND_TO_IDX[h] == i


def test_hand_contains_card_shape():
    assert HAND_CONTAINS_CARD.shape == (52, 1326)


def test_hand_contains_card_correct():
    # hand index 0 is (0, 1): cards 0 and 1
    assert HAND_CONTAINS_CARD[0][0] == True
    assert HAND_CONTAINS_CARD[1][0] == True
    assert HAND_CONTAINS_CARD[2][0] == False


def test_bucket_count():
    assert len(set(HAND_BUCKET.tolist())) == 169


def test_pocket_aces_bucket():
    # AA: rank 12, all 6 combos should map to bucket 12 (top pair bucket)
    aa_hands = [(c1, c2) for (c1, c2) in ALL_HANDS
                if c1 // 4 == 12 and c2 // 4 == 12]
    assert len(aa_hands) == 6
    for h in aa_hands:
        idx = HAND_TO_IDX[h]
        assert HAND_BUCKET[idx] == 12, f"AA not in pair bucket: {h}"


def test_aks_bucket():
    # AKs: r_hi=12 (A), r_lo=11 (K), suited — should be bucket 13 (first suited bucket)
    # Find an AKs hand. A spades = 12*4+3=51, K spades = 11*4+3=47
    ac_idx = 12 * 4 + 3  # As = 51
    kc_idx = 11 * 4 + 3  # Ks = 47
    h = (min(ac_idx, kc_idx), max(ac_idx, kc_idx))
    idx = HAND_TO_IDX[h]
    bucket = HAND_BUCKET[idx]
    assert bucket == 13, f"AKs should be bucket 13 (first suited), got {bucket}"
