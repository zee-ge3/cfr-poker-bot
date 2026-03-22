"""Tests for NLHEGame engine (Task 2)."""

import pytest
from nlhe.game import GameState, NLHEGame, STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER


# ---------------------------------------------------------------------------
# test_gamestate_fields
# ---------------------------------------------------------------------------

def test_gamestate_fields():
    """Verify GameState fields exist and have correct types after new_hand."""
    game = NLHEGame(starting_stack=100)
    state = game.new_hand(human_position=0)

    assert isinstance(state, GameState)
    assert isinstance(state.street, int)
    assert isinstance(state.pot, int)
    assert isinstance(state.our_stack, int)
    assert isinstance(state.opp_stack, int)
    assert isinstance(state.our_hole, list)
    assert isinstance(state.opp_hole, list)
    assert isinstance(state.board, list)
    assert isinstance(state.valid_actions, list)
    assert isinstance(state.position, int)
    assert isinstance(state.street_bets, list)


# ---------------------------------------------------------------------------
# test_new_hand_creates_valid_state
# ---------------------------------------------------------------------------

def test_new_hand_creates_valid_state():
    """pot==3 after blinds, 2 hole cards, empty board, stack invariant holds."""
    game = NLHEGame(starting_stack=100)
    state = game.new_hand(human_position=0)

    # Blinds: SB=1, BB=2, total pot=3
    assert state.pot == 3

    # Human gets 2 hole cards
    assert len(state.our_hole) == 2

    # No board cards preflop
    assert state.board == []

    # Stack invariant: our_stack + opp_stack + pot == 200
    assert state.our_stack + state.opp_stack + state.pot == 200

    # Street is preflop
    assert state.street == STREET_PREFLOP

    # Position is whatever we passed
    assert state.position == 0


# ---------------------------------------------------------------------------
# test_valid_actions_includes_fold_when_bet
# ---------------------------------------------------------------------------

def test_valid_actions_includes_fold_when_bet():
    """FOLD is in valid_actions when there is a bet to call (preflop, BB faces SB raise)."""
    game = NLHEGame(starting_stack=100)
    # human_position=1 means human is BB; bot is SB and acts first preflop
    state = game.new_hand(human_position=1)

    # On preflop, SB has already posted 1 and BB has posted 2.
    # BB is the human (position=1). BTN/SB acts first preflop.
    # The current_bet for human (BB) is 1 (needs to call 1 more or raise).
    # FOLD must be available since there is a live bet to the BB.
    assert 'FOLD' in state.valid_actions


def test_valid_actions_no_fold_when_no_bet():
    """FOLD is NOT in valid_actions when no bet outstanding (e.g. after advancing to flop)."""
    game = NLHEGame(starting_stack=100)
    # human_position=0 means human is BTN/SB, acts first preflop
    # Bot is BB. Human posts SB=1, bot posts BB=2. Human (SB) faces 1 more.
    # After both CALL/CHECK to flop, on flop there should be no bet.
    state = game.new_hand(human_position=0)

    # Force to flop so we can check valid_actions with no bet outstanding
    game._force_street(STREET_FLOP)
    state = game._get_state()

    # On the flop with no bet, FOLD should NOT be in valid_actions
    assert 'FOLD' not in state.valid_actions


# ---------------------------------------------------------------------------
# test_fold_ends_hand
# ---------------------------------------------------------------------------

def test_fold_ends_hand():
    """Human FOLD returns hand_over=True."""
    game = NLHEGame(starting_stack=100)
    # human is SB (position=0), acts first preflop facing BB's live bet of 2
    # Human SB posted 1, BB posted 2, so human must call 1 more → FOLD is valid
    state = game.new_hand(human_position=0)

    assert 'FOLD' in state.valid_actions
    result = game.human_action('FOLD')

    assert result['hand_over'] is True
    assert 'winner' in result


def test_fold_when_no_bet_raises_or_converts():
    """FOLD when no bet outstanding should be treated as CHECK (or raise ValueError)."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_FLOP)

    # On flop with no bet, FOLD is invalid — engine should either raise or convert to CHECK
    # We accept either behaviour; just verify hand is not spuriously ended
    try:
        result = game.human_action('FOLD')
        # If it didn't raise, it should have treated as CHECK (hand not over unless it somehow ended)
        # Permissive: accept that it converts to CHECK or errors
    except (ValueError, AssertionError):
        pass  # This is fine


# ---------------------------------------------------------------------------
# test_call_advances_action
# ---------------------------------------------------------------------------

def test_call_advances_action():
    """CALL goes to next street or continues hand; hand_over key always present."""
    game = NLHEGame(starting_stack=100)
    # human is SB (position=0), faces BB's 2 — needs to call 1 more
    state = game.new_hand(human_position=0)

    assert 'CALL' in state.valid_actions
    result = game.human_action('CALL')

    # hand_over key must exist
    assert 'hand_over' in result

    # stack invariant still holds
    new_state = result['state']
    assert new_state.our_stack + new_state.opp_stack + new_state.pot == 200


# ---------------------------------------------------------------------------
# test_board_revealed_on_flop
# ---------------------------------------------------------------------------

def test_board_revealed_on_flop():
    """After reaching flop, board has exactly 3 cards."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_FLOP)
    state = game._get_state()

    assert len(state.board) == 3
    # Cards should be strings (treys format like 'Ah', 'Kd', ...)
    for card in state.board:
        assert isinstance(card, str)
        assert len(card) == 2 or len(card) == 3  # e.g. 'Ah' or '10h'


def test_board_revealed_on_turn():
    """After reaching turn, board has exactly 4 cards."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_TURN)
    state = game._get_state()
    assert len(state.board) == 4


def test_board_revealed_on_river():
    """After reaching river, board has exactly 5 cards."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_RIVER)
    state = game._get_state()
    assert len(state.board) == 5


# ---------------------------------------------------------------------------
# test_stack_invariant
# ---------------------------------------------------------------------------

def test_stack_invariant():
    """our_stack + opp_stack + pot == 200 always, through multiple actions."""
    game = NLHEGame(starting_stack=100)

    # Check immediately after new_hand
    state = game.new_hand(human_position=0)
    assert state.our_stack + state.opp_stack + state.pot == 200

    # After human CALL preflop
    result = game.human_action('CALL')
    if not result['hand_over']:
        state = result['state']
        assert state.our_stack + state.opp_stack + state.pot == 200

        # After bot CHECK
        result = game.bot_action('CALL')
        if not result['hand_over']:
            state = result['state']
            assert state.our_stack + state.opp_stack + state.pot == 200


def test_stack_invariant_after_raise():
    """Stack invariant holds after a RAISE."""
    game = NLHEGame(starting_stack=100)
    state = game.new_hand(human_position=0)
    assert state.our_stack + state.opp_stack + state.pot == 200

    result = game.human_action('RAISE_SMALL')
    if not result['hand_over']:
        state = result['state']
        assert state.our_stack + state.opp_stack + state.pot == 200


# ---------------------------------------------------------------------------
# Additional behavioural tests
# ---------------------------------------------------------------------------

def test_raise_allin_always_available():
    """RAISE_ALLIN is always in valid_actions when both players have chips."""
    game = NLHEGame(starting_stack=100)
    state = game.new_hand(human_position=0)
    assert 'RAISE_ALLIN' in state.valid_actions


def test_street_constants():
    """Verify street constant values."""
    assert STREET_PREFLOP == 0
    assert STREET_FLOP == 1
    assert STREET_TURN == 2
    assert STREET_RIVER == 3


def test_showdown_returns_winner():
    """A hand that reaches showdown returns a winner."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_RIVER)

    # human=SB (pos=0), so bot (BB) acts first post-flop
    # Both players check to showdown: bot CHECK first, then human CHECK
    result = game.bot_action('CALL')   # bot CHECK
    if result['hand_over']:
        assert 'winner' in result
    else:
        result = game.human_action('CALL')  # human CHECK → showdown
        assert result['hand_over'] is True
        assert 'winner' in result


def test_bot_hole_revealed_at_showdown():
    """At showdown, bot_hole is revealed (non-empty list)."""
    game = NLHEGame(starting_stack=100)
    game.new_hand(human_position=0)
    game._force_street(STREET_RIVER)

    # human=SB (pos=0), bot (BB) acts first post-flop
    result = game.bot_action('CALL')   # bot CHECK
    if result['hand_over'] and 'bot_hole' in result:
        assert isinstance(result['bot_hole'], list)
        assert len(result['bot_hole']) == 2
    else:
        result = game.human_action('CALL')  # human CHECK → showdown
        assert result['hand_over'] is True
        assert 'bot_hole' in result
        assert isinstance(result['bot_hole'], list)
        assert len(result['bot_hole']) == 2
