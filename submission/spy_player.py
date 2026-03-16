# submission/spy_player.py
"""
SpyAgent — data-extraction bot. Submitted temporarily to collect opponent intelligence.
Sole goal: maximize information yield, not chips.
"""
import random

from agents.agent import Agent
from gym_env import PokerEnv

AT = PokerEnv.ActionType


class PlayerAgent(Agent):
    """Named PlayerAgent so tournament server picks it up correctly."""

    def __name__(self):
        return "SpyAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._hand_mode = None
        self._prev_street = -1
        self._hand_count = 0

    def _reset_hand_state(self):
        self._hand_mode = None
        self._prev_street = -1

    def _select_mode(self, obs) -> str:
        in_position = obs.get("blind_position", 0) == 0
        base = 'raise' if in_position else 'call'
        if random.random() < 0.20:
            base = 'call' if base == 'raise' else 'raise'
        return base

    def act(self, obs, reward, terminated, truncated, info):
        # observe() handles _hand_count increment — do NOT increment here
        if terminated:
            return (AT.FOLD.value, 0, 0, 0)

        valid = obs["valid_actions"]
        street = obs["street"]

        if valid[AT.DISCARD.value]:
            return (AT.DISCARD.value, 0, 0, 1)

        if self._hand_mode is None:
            self._hand_mode = self._select_mode(obs)

        self._prev_street = street
        self.logger.info(
            f"H{self._hand_count} s{street} mode={self._hand_mode} "
            f"IP={obs.get('blind_position',0)==0}"
        )

        if self._hand_mode == 'raise':
            return self._raise_action(obs)
        return self._call_action(obs)

    def _raise_action(self, obs):
        valid = obs["valid_actions"]
        if valid[AT.FOLD.value] and random.random() < 0.10:
            return (AT.FOLD.value, 0, 0, 0)
        if valid[AT.RAISE.value]:
            pot = obs["my_bet"] + obs["opp_bet"]
            frac = random.uniform(0.3, 1.2)
            amount = max(obs["min_raise"], min(obs["max_raise"], int(pot * frac)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def _call_action(self, obs):
        valid = obs["valid_actions"]
        if valid[AT.RAISE.value] and random.random() < 0.10:
            pot = obs["my_bet"] + obs["opp_bet"]
            amount = max(obs["min_raise"], min(obs["max_raise"], int(pot * 0.5)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def observe(self, obs, reward, terminated, truncated, info):
        """Single increment point for _hand_count."""
        if terminated:
            self._hand_count += 1
            self._reset_hand_state()
