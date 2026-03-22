"""
Match Manager: tracks match state and provides continuous phase signals.
All transitions are smooth and jittered — no hard phase boundaries.
"""
import math
import random


class MatchManager:
    TOTAL_HANDS = 1000
    BLEED_RATE = 1.5

    def __init__(self):
        self._hand_number = 0
        self._cumulative_reward = 0
        self._time_left = 1500.0
        self._time_used = 0.0

    def update(self, reward: float, time_used: float = 0.0, time_left: float = 0.0):
        self._hand_number += 1
        self._cumulative_reward += reward
        if time_left > 0:
            self._time_left = time_left
        if time_used > 0:
            self._time_used = time_used

    def get_state(self) -> dict:
        remaining = max(1, self.TOTAL_HANDS - self._hand_number)
        remaining_bleed = remaining * self.BLEED_RATE

        if remaining_bleed > 0:
            pressure = -self._cumulative_reward / remaining_bleed
            pressure = max(-1.0, min(1.0, pressure))
        else:
            pressure = 0.0

        # Aggression is a gentle nudge, not a sledgehammer.
        # Only apply mild loosening when behind, capped at 0.25.
        # No early_factor — playing loose early is just gambling.
        aggression = max(0.0, pressure) * 0.25
        aggression = min(0.25, aggression)
        aggression *= (1.0 + random.uniform(-0.10, 0.10))
        aggression = max(0.0, min(0.25, aggression))

        in_protection = (self._cumulative_reward > remaining_bleed * 0.15
                         and self._hand_number > 50)

        # Lockout: guaranteed win. Bleed rate 1.5/hand (avg SB+BB cost).
        # Buffer of 2 chips (one big blind). No hand minimum — lock out ASAP.
        lockout_threshold = remaining * self.BLEED_RATE + 2
        in_lockout = self._cumulative_reward > lockout_threshold

        if in_protection:
            aggression *= 0.3

        return {
            'hand_number': self._hand_number,
            'cumulative_reward': self._cumulative_reward,
            'remaining_hands': remaining,
            'pressure': pressure,
            'aggression': aggression,
            'in_protection': in_protection,
            'in_lockout': in_lockout,
            'time_left': self._time_left,
            'time_per_hand': self._time_left / max(1, remaining),
        }

    @property
    def hands_remaining(self) -> int:
        return max(1, self.TOTAL_HANDS - self._hand_number)

    def get_strategy_mode(self) -> str:
        """
        Determine action-selection mode based on match state.

        Returns:
            'protection' — significant lead, over halfway through match
            'pressure'   — significant deficit
            'neutral'    — default
        """
        margin = self._cumulative_reward
        hands_left = self.hands_remaining
        if margin > 200:
            return 'protection'
        elif margin < -200:
            return 'pressure'
        return 'neutral'

