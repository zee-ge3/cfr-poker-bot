"""SpyAgent — data-extraction bot for CMU AI Poker Tournament.

Strategy: collect opponent intelligence, not chips.
- IP (blind_position=0, SB, acts last post-flop): raise-mode — probe every street
- OOP (blind_position=1, BB, acts first post-flop): call-mode — see showdowns
- 20% mode-flip randomization prevents opponent anti-modeling
"""
import random
from agents.agent import Agent
from gym_env import PokerEnv

AT = PokerEnv.ActionType


class SpyAgent(Agent):

    def __name__(self):
        return "SpyAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._hand_mode = None   # 'raise' | 'call', set on first act() of each hand
        self._prev_street = -1
        self._hand_count = 0

    def _reset_hand_state(self):
        """Reset per-hand state. Called at hand start."""
        self._hand_mode = None
        self._prev_street = -1

    def _select_mode(self, obs) -> str:
        """Choose raise or call mode for this hand based on position."""
        in_position = obs.get("blind_position", 0) == 0   # SB = IP post-flop
        base = 'raise' if in_position else 'call'
        if random.random() < 0.20:
            base = 'call' if base == 'raise' else 'raise'
        return base

    def act(self, obs, reward, terminated, truncated, info):
        # Note: do NOT increment _hand_count here. observe() handles end-of-hand
        # bookkeeping (same pattern as player.py). act() with terminated=True is
        # a terminal signal but observe() is always called after it.
        if terminated:
            return (AT.FOLD.value, 0, 0, 0)

        valid = obs["valid_actions"]
        street = obs["street"]

        # Discard: always keep first two cards (indices 0 and 1)
        if valid[AT.DISCARD.value]:
            return (AT.DISCARD.value, 0, 0, 1)

        # Set mode at the first action of each new hand
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
        """Raise mode: probe every street with random sizing."""
        valid = obs["valid_actions"]
        # 10% chance to fold instead (disguise — looks like a weak player)
        if valid[AT.FOLD.value] and random.random() < 0.10:
            return (AT.FOLD.value, 0, 0, 0)

        if valid[AT.RAISE.value]:
            pot = obs["my_bet"] + obs["opp_bet"]
            min_r = obs["min_raise"]
            max_r = obs["max_raise"]
            frac = random.uniform(0.3, 1.2)
            amount = max(min_r, min(max_r, int(pot * frac)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def _call_action(self, obs):
        """Call mode: see showdown to collect hand-strength data."""
        valid = obs["valid_actions"]
        # 10% IP-limp disguise: occasionally raise even in call mode
        if valid[AT.RAISE.value] and random.random() < 0.10:
            pot = obs["my_bet"] + obs["opp_bet"]
            min_r = obs["min_raise"]
            max_r = obs["max_raise"]
            amount = max(min_r, min(max_r, int(pot * 0.5)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def observe(self, obs, reward, terminated, truncated, info):
        """Handles end-of-hand bookkeeping (single increment point for _hand_count)."""
        if terminated:
            self._hand_count += 1
            self._reset_hand_state()
