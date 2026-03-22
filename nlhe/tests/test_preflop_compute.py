"""Unit tests for nlhe.cfr.preflop_compute."""

import pytest
from nlhe.cfr.preflop_compute import _fold_ev, STARTING_STACK, SB_POST, BB_POST


# ---------------------------------------------------------------------------
# _fold_ev correctness
# ---------------------------------------------------------------------------

def test_fold_ev_sb_folds_initial():
    """SB folds before investing any chips beyond SB_POST.
    SB has posted 1 chip, so stack_sb = 99, pot = 3.
    EV = -(100 - 99) = -1.
    """
    stack_sb = STARTING_STACK - SB_POST  # 99
    stack_bb = STARTING_STACK - BB_POST  # 98
    pot = SB_POST + BB_POST              # 3
    ev = _fold_ev(pot, stack_sb, stack_bb, folder=0)
    assert ev == pytest.approx(-1.0), f"Expected -1.0, got {ev}"


def test_fold_ev_bb_folds_to_limp():
    """BB folds to SB's CALL (limp).
    SB invested 1 chip (call cost) + 1 (SB_POST) = total 1 chip beyond start? No:
    SB posted SB_POST=1, then calls 1 more -> stack_sb = 98, pot = 4.
    But if we test the scenario described: SB limps (calls BB), so
    stack_sb = 98, pot = 4. BB folds.
    EV = pot - (STARTING_STACK - stack_sb) = 4 - (100 - 98) = 4 - 2 = +2.
    SB wins BB's 2 chips (net +2 relative to their starting 100 chips).
    """
    stack_sb = STARTING_STACK - BB_POST  # 98 (SB called, now equal to BB's post)
    stack_bb = STARTING_STACK - BB_POST  # 98
    pot = BB_POST * 2                    # 4 (both in 2)
    ev = _fold_ev(pot, stack_sb, stack_bb, folder=1)
    assert ev == pytest.approx(2.0), f"Expected +2.0, got {ev}"


def test_fold_ev_bb_folds_to_allin():
    """BB folds to SB's raise all-in.
    SB pushed all 100 chips in: stack_sb = 0, pot = 100 + 2 = 102.
    EV = pot - (STARTING_STACK - stack_sb) = 102 - 100 = +2.
    SB wins BB's blind (2 chips net).
    """
    stack_sb = 0.0
    stack_bb = STARTING_STACK - BB_POST  # 98
    pot = STARTING_STACK + BB_POST       # 102
    ev = _fold_ev(pot, stack_sb, stack_bb, folder=1)
    assert ev == pytest.approx(2.0), f"Expected +2.0, got {ev}"


def test_fold_ev_bb_folds_to_small_raise():
    """BB folds to SB's small raise.
    SB posted 1, then raises 5 more -> stack_sb = 94, pot = 3 + 5 = 8.
    EV = pot - invested_sb = 8 - (100 - 94) = 8 - 6 = +2.
    """
    stack_sb = 94.0
    stack_bb = STARTING_STACK - BB_POST  # 98
    pot = 8.0
    ev = _fold_ev(pot, stack_sb, stack_bb, folder=1)
    assert ev == pytest.approx(2.0), f"Expected +2.0 (always wins BB blind net), got {ev}"


def test_fold_ev_bb_folds_net_equals_bb_post():
    """When BB folds, SB always nets +BB_POST regardless of raise size.
    Because pot = invested_sb + BB_POST, so pot - invested_sb = BB_POST.
    """
    for sb_invest in [1.0, 3.0, 10.0, 50.0, 100.0]:
        stack_sb = STARTING_STACK - sb_invest
        # pot includes SB's investment + BB's blind
        pot = sb_invest + BB_POST
        ev = _fold_ev(pot, stack_sb, 0.0, folder=1)
        assert ev == pytest.approx(BB_POST), (
            f"Expected +{BB_POST} for sb_invest={sb_invest}, got {ev}"
        )
