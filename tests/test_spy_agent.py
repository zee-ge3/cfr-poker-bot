import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.spy_agent import SpyAgent
from gym_env import PokerEnv

AT = PokerEnv.ActionType

def make_obs(blind_position=0, street=0, valid=None, my_bet=1, opp_bet=1,
             min_raise=2, max_raise=200, my_cards=None, opp_discarded=None,
             community=None, opp_last=""):
    if valid is None:
        valid = [False] * 6
        valid[AT.RAISE.value] = True
        valid[AT.CALL.value] = True
        valid[AT.CHECK.value] = True
    return {
        "blind_position": blind_position,
        "street": street,
        "valid_actions": valid,
        "my_bet": my_bet,
        "opp_bet": opp_bet,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "my_cards": my_cards or [0, 1],
        "opp_discarded_cards": opp_discarded or [-1, -1, -1],
        "community_cards": community or [-1, -1, -1, -1, -1],
        "opp_last_action": opp_last,
        "time_used": 0.0,
        "time_left": 500.0,
    }


def test_ip_defaults_to_raise_mode():
    """IP (blind_position=0) bot raises pre-flop at least 70% of the time."""
    import random
    random.seed(42)
    agent = SpyAgent(stream=False)
    raise_count = 0
    n = 200
    for _ in range(n):
        agent._reset_hand_state()
        obs = make_obs(blind_position=0, street=0)
        action = agent.act(obs, 0, False, False, {})
        if action[0] == AT.RAISE.value:
            raise_count += 1
    # Expect ≥70% raise (80% base IP raise mode, 20% flip to call mode)
    assert raise_count / n >= 0.70, f"IP raise rate {raise_count/n:.2%} < 70%"


def test_oop_defaults_to_call_mode():
    """OOP (blind_position=1) bot calls/checks pre-flop at least 70% of the time."""
    import random
    random.seed(99)
    agent = SpyAgent(stream=False)
    call_count = 0
    n = 200
    for _ in range(n):
        agent._reset_hand_state()
        obs = make_obs(blind_position=1, street=0)
        action = agent.act(obs, 0, False, False, {})
        if action[0] in (AT.CALL.value, AT.CHECK.value):
            call_count += 1
    assert call_count / n >= 0.70, f"OOP call/check rate {call_count/n:.2%} < 70%"
