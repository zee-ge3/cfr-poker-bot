import pytest
from nlhe.bot import Bot
from nlhe.game import GameState, STREET_PREFLOP, STREET_FLOP


def test_bot_new_hand():
    bot = Bot()
    bot.new_hand(hole_cards=['As', 'Kd'], position=0)
    assert bot._hole_cards == ['As', 'Kd']
    assert bot._position == 0


def test_bot_decide_preflop_returns_action():
    bot = Bot()
    bot.new_hand(hole_cards=['As', 'Kd'], position=0)
    state = GameState(
        street=STREET_PREFLOP, pot=3, our_stack=99, opp_stack=98,
        our_hole=['As', 'Kd'], opp_hole=[], board=[],
        valid_actions=['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN'],
        position=0, street_bets=[1, 2],
    )
    action = bot.decide(state)
    assert action in state.valid_actions


def test_bot_decide_postflop_returns_action():
    bot = Bot()
    bot.new_hand(hole_cards=['As', 'Ac'], position=0)
    state = GameState(
        street=STREET_FLOP, pot=10, our_stack=95, opp_stack=95,
        our_hole=['As', 'Ac'], opp_hole=[], board=['Kd', '5s', '2c'],
        valid_actions=['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_ALLIN'],
        position=0, street_bets=[],
    )
    action = bot.decide(state)
    assert action in state.valid_actions


def test_bot_observe_action_does_not_crash():
    bot = Bot()
    bot.new_hand(hole_cards=['As', 'Ac'], position=0)
    state = GameState(
        street=STREET_FLOP, pot=10, our_stack=95, opp_stack=95,
        our_hole=['As', 'Ac'], opp_hole=[], board=['Kd', '5s', '2c'],
        valid_actions=['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_ALLIN'],
        position=0, street_bets=[],
    )
    bot.observe_action('RAISE_ALLIN', 95)


def test_no_gym_env_import():
    import importlib
    import sys
    # Importing bot should not pull in gym_env
    if 'nlhe.bot' in sys.modules:
        del sys.modules['nlhe.bot']
    mod = importlib.import_module('nlhe.bot')
    assert 'gym_env' not in str(mod.__file__)
