"""Tests for nlhe.cfr.solver — NLHESolver (52-card NLHE CFR-D port)."""

import numpy as np
import pytest

from nlhe.game import GameState, STREET_RIVER, STREET_FLOP, STREET_PREFLOP


def make_river_state(pot=20, stack=80):
    """Build a river GameState with AA (near-nut hand) for testing."""
    return GameState(
        street=STREET_RIVER,
        pot=pot,
        our_stack=stack,
        opp_stack=stack,
        our_hole=['As', 'Ac'],
        opp_hole=[],
        board=['Kd', '5s', '2c', '7h', 'Qh'],
        valid_actions=['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN'],
        position=0,
        street_bets=[],
    )


def make_flop_state(pot=10, stack=90):
    """Build a flop GameState for testing."""
    return GameState(
        street=STREET_FLOP,
        pot=pot,
        our_stack=stack,
        opp_stack=stack,
        our_hole=['As', 'Ac'],
        opp_hole=[],
        board=['Kd', '5s', '2c'],
        valid_actions=['CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN'],
        position=0,
        street_bets=[],
    )


def make_preflop_state(pot=3, stack=97):
    """Build a preflop GameState for testing."""
    return GameState(
        street=STREET_PREFLOP,
        pot=pot,
        our_stack=stack,
        opp_stack=stack,
        our_hole=['As', 'Ac'],
        opp_hole=[],
        board=[],
        valid_actions=['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN'],
        position=0,
        street_bets=[],
    )


# ---------------------------------------------------------------------------
# Test 1: solver initializes without crashing
# ---------------------------------------------------------------------------

def test_solver_initializes():
    """NLHESolver(state, our_hole) must not raise an exception."""
    from nlhe.cfr.solver import NLHESolver

    state = make_river_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.1)

    # Basic sanity: opp_range sums to 1.0
    assert solver.opp_range is not None
    assert solver.opp_range.shape == (1326,)
    assert abs(solver.opp_range.sum() - 1.0) < 1e-4, (
        f"opp_range should sum to 1.0, got {solver.opp_range.sum()}"
    )

    # Our hole cards are dead, so range[our_hand_idx] should be 0
    from nlhe.cfr.abstraction import HAND_TO_IDX
    from nlhe.cfr.equity import card_str_to_idx
    our_idx_tuple = tuple(sorted(card_str_to_idx(c) for c in ['As', 'Ac']))
    our_hand_idx = HAND_TO_IDX.get(our_idx_tuple, None)
    if our_hand_idx is not None:
        assert solver.opp_range[our_hand_idx] == 0.0, (
            "opp_range should be 0 for our own hole card combo"
        )


# ---------------------------------------------------------------------------
# Test 2: solver returns a valid action
# ---------------------------------------------------------------------------

def test_solver_returns_valid_action():
    """solve() must return a string that is in state.valid_actions."""
    from nlhe.cfr.solver import NLHESolver

    state = make_river_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.2)
    action = solver.solve()

    assert isinstance(action, str), f"solve() returned non-string: {action!r}"
    assert action in state.valid_actions, (
        f"solve() returned {action!r} not in valid_actions {state.valid_actions}"
    )


def test_solver_returns_valid_action_flop():
    """solve() must return valid action on a flop board too."""
    from nlhe.cfr.solver import NLHESolver

    state = make_flop_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.2)
    action = solver.solve()

    assert action in state.valid_actions, (
        f"Flop: solve() returned {action!r} not in {state.valid_actions}"
    )


def test_solver_returns_valid_action_preflop():
    """solve() must return valid action on preflop."""
    from nlhe.cfr.solver import NLHESolver

    state = make_preflop_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.2)
    action = solver.solve()

    assert action in state.valid_actions, (
        f"Preflop: solve() returned {action!r} not in {state.valid_actions}"
    )


# ---------------------------------------------------------------------------
# Test 3: strategy is stable after enough iterations
# ---------------------------------------------------------------------------

def test_solver_strategy_is_stable_after_enough_iterations():
    """After a longer budget, the average strategy should concentrate on some action.

    Checks that at least one action has avg probability > 0.2, i.e. the solver
    doesn't remain stuck at a degenerate uniform distribution.
    """
    from nlhe.cfr.solver import NLHESolver

    state = make_river_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=1.0)
    solver.solve()

    assert solver._avg_strategy is not None, "_avg_strategy should be set after solve()"
    assert solver._avg_strategy.shape == (len(state.valid_actions),), (
        f"Expected shape ({len(state.valid_actions)},), got {solver._avg_strategy.shape}"
    )

    # Should sum to approximately 1.0
    total = solver._avg_strategy.sum()
    assert abs(total - 1.0) < 1e-3, f"avg_strategy should sum to 1.0, got {total}"

    # At least one action should dominate (not stuck uniform over 5 actions = 0.2)
    max_prob = solver._avg_strategy.max()
    assert max_prob > 0.2, (
        f"Expected max strategy prob > 0.2 (not stuck uniform), got {max_prob:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: near-nut hand on river does not fold
# ---------------------------------------------------------------------------

def test_nuts_does_not_fold():
    """AA on a Kd 5s 2c 7h Qh board is a strong hand — solver should not FOLD."""
    from nlhe.cfr.solver import NLHESolver

    state = make_river_state(pot=20, stack=80)
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.5)
    action = solver.solve()

    assert action != 'FOLD', (
        f"Solver folded a near-nut hand (AA) on river — action was {action!r}"
    )


def test_nuts_does_not_fold_larger_pot():
    """AA with a larger pot (more at stake) should still not fold."""
    from nlhe.cfr.solver import NLHESolver

    state = make_river_state(pot=60, stack=40)
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.5)
    action = solver.solve()

    assert action != 'FOLD', (
        f"Solver folded AA on river with pot=60 — action was {action!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: observe_action updates opp_range
# ---------------------------------------------------------------------------

def test_solver_observe_action_updates_range():
    """observe_action should change opp_range (not leave it unchanged)."""
    from nlhe.cfr.solver import NLHESolver

    state = make_flop_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.2)
    solver.solve()  # need avg_strategy to be set

    range_before = solver.opp_range.copy()

    # Observing a raise action should update the range
    solver.observe_action('RAISE_LARGE')

    range_after = solver.opp_range.copy()

    # The range should have changed (not identical to before)
    assert not np.allclose(range_before, range_after), (
        "opp_range did not change after observe_action — range update not working"
    )

    # The updated range should still sum to ~1.0
    total = range_after.sum()
    assert abs(total - 1.0) < 1e-4, (
        f"opp_range after observe_action should sum to 1.0, got {total:.6f}"
    )


def test_solver_observe_action_without_solve():
    """observe_action before solve() (no avg_strategy) should apply dead-card zeroing only."""
    from nlhe.cfr.solver import NLHESolver

    state = make_flop_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.1)
    # Do NOT call solve() — _avg_strategy is None

    range_before = solver.opp_range.copy()

    # Should not crash even without avg_strategy
    solver.observe_action('CALL')

    range_after = solver.opp_range.copy()

    # Range should still be valid (sum to ~1.0)
    total = range_after.sum()
    assert abs(total - 1.0) < 1e-4, (
        f"opp_range after observe_action (no solve) should sum to 1.0, got {total:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6: single-action early return path
# ---------------------------------------------------------------------------

def test_solver_single_action_returns_it():
    """When only one valid action exists, solve() returns it without running CFR."""
    from nlhe.cfr.solver import NLHESolver

    state = GameState(
        street=STREET_RIVER,
        pot=20,
        our_stack=80,
        opp_stack=80,
        our_hole=['As', 'Ac'],
        opp_hole=[],
        board=['Kd', '5s', '2c', '7h', 'Qh'],
        valid_actions=['CALL'],  # only one valid action
        position=0,
        street_bets=[],
    )
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.1)
    action = solver.solve()
    assert action == 'CALL'
    assert solver._avg_strategy is not None
    assert solver._avg_strategy.shape == (1,)
    assert abs(solver._avg_strategy[0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 7: observe_action called twice sequentially stays valid
# ---------------------------------------------------------------------------

def test_solver_observe_action_twice_stays_valid():
    """Two sequential observe_action calls should not collapse the range."""
    from nlhe.cfr.solver import NLHESolver

    state = make_flop_state()
    solver = NLHESolver(state, our_hole=['As', 'Ac'], budget_seconds=0.2)
    solver.solve()

    solver.observe_action('RAISE_LARGE')
    solver.observe_action('CALL')

    total = solver.opp_range.sum()
    assert abs(total - 1.0) < 1e-4, f"opp_range after 2 observe_action calls should sum to 1.0, got {total:.6f}"
    # Range should still have some live entries
    assert (solver.opp_range > 0).any(), "opp_range should not be all-zero after 2 observe_action calls"
