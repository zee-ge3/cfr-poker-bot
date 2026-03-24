"""
Opponent model — tracks behavior across hands for range narrowing and exploitation.

Ported from trips_poker competition code. Tracks VPIP, fold rate, bluff rate,
and provides continuous range weighting based on opponent selectivity.
"""
from __future__ import annotations

import numpy as np

# Confidence fully saturates after this many hands
_CONFIDENCE_HANDS = 50


class OpponentModel:
    """
    Track opponent behavior across multiple hands.

    Provides:
    - Action frequency stats (VPIP, fold rate, bluff rate)
    - Per-action range weights for Bayesian narrowing
    - Continuous range weighting based on opponent selectivity
    """

    def __init__(self) -> None:
        self._hands_observed: int = 0
        self._raises: int = 0
        self._calls: int = 0
        self._checks: int = 0
        self._folds: int = 0
        self._total_actions: int = 0

        # Fold response to our raises
        self._folds_to_our_raise: int = 0
        self._faces_our_raise: int = 0

        # Bayesian bluff rate (Beta distribution)
        self._bluff_alpha: float = 1.0
        self._bluff_beta: float = 3.0

    def new_hand(self) -> None:
        """Call at start of each new hand."""
        self._hands_observed += 1

    def observe_action(self, action: str, street: int) -> None:
        """Record an opponent action."""
        self._total_actions += 1
        if action.startswith("RAISE"):
            self._raises += 1
            if street >= 2:
                self._bluff_alpha += 0.3
        elif action == "CALL":
            self._calls += 1
        elif action == "FOLD":
            self._folds += 1
        else:
            self._checks += 1

    def observe_response_to_our_raise(self, action: str) -> None:
        """Record how opponent responded when we raised."""
        self._faces_our_raise += 1
        if action == "FOLD":
            self._folds_to_our_raise += 1

    def get_context(self) -> dict:
        """Return current opponent profile."""
        conf = self._confidence()
        prior_vpip = 0.5
        prior_fold = 0.3

        if self._total_actions > 0:
            raw_vpip = (self._total_actions - self._folds) / self._total_actions
        else:
            raw_vpip = prior_vpip

        if self._faces_our_raise > 0:
            raw_fold = self._folds_to_our_raise / self._faces_our_raise
        else:
            raw_fold = prior_fold

        vpip = prior_vpip * (1 - conf) + raw_vpip * conf
        fold_rate = prior_fold * (1 - conf) + raw_fold * conf
        bluff_rate = self._bluff_alpha / (self._bluff_alpha + self._bluff_beta)

        return {
            "hands_observed": self._hands_observed,
            "vpip": vpip,
            "fold_rate": fold_rate,
            "bluff_rate": bluff_rate,
            "raise_rate": self._raises / max(self._total_actions, 1),
            "confidence": conf,
        }

    def get_range_weights(
        self, hand_strengths: np.ndarray, action: str
    ) -> np.ndarray:
        """
        Bayesian range update weights for a given action.

        Parameters
        ----------
        hand_strengths : (1326,) float32 hand strength percentiles [0, 1]
        action : action string

        Returns
        -------
        (1326,) float32 multiplicative likelihood weights
        """
        s = hand_strengths
        if action == "RAISE_ALLIN":
            # Very polarized: strong hands and a few bluffs (lowest strengths)
            # Power 4 sharply penalizes medium/weak hands
            weights = np.power(s, 4.0)
            # Add a small bluffing tail (e.g. missed draws)
            bluffs = np.power(1.0 - s, 6.0) * 0.1
            return (weights + bluffs).astype(np.float32).clip(min=0.001)
        elif action in ("RAISE_LARGE", "RAISE_OVERBET"):
            # Strong hands mostly, power 2.5
            weights = np.power(s, 2.5)
            bluffs = np.power(1.0 - s, 4.0) * 0.15
            return (weights + bluffs).astype(np.float32).clip(min=0.01)
        elif action == "RAISE_SMALL":
            # Value hands and semi-bluffs
            weights = np.power(s, 1.5)
            bluffs = np.power(1.0 - s, 2.0) * 0.2
            return (weights + bluffs).astype(np.float32).clip(min=0.02)
        elif action == "CALL":
            # Medium strength hands (too weak to raise, too strong to fold)
            # Peak around 0.6-0.8
            # A bell curve: exp(-((s - 0.7) / 0.2)^2)
            return np.exp(-np.power((s - 0.7) / 0.25, 2.0)).astype(np.float32).clip(min=0.05)
        elif action == "FOLD":
            # Weak hands
            return np.power(1.0 - s, 3.0).astype(np.float32).clip(min=0.01)
        else:
            # CHECK: usually weak or medium, sometimes slowplaying
            return (0.7 * (1.0 - s) + 0.3).astype(np.float32)

    def apply_continuous_range_weighting(
        self, opp_range: np.ndarray, hand_strengths: np.ndarray
    ) -> np.ndarray:
        """
        Shift opponent range based on how selective they are (VPIP).

        Tight opponents (low VPIP) have ranges concentrated on strong hands.
        Loose opponents leave range mostly unchanged.
        """
        ctx = self.get_context()
        conf = ctx["confidence"]
        vpip = ctx["vpip"]

        selectivity = max(0.0, 1.0 - vpip) * conf

        if selectivity < 0.01:
            return opp_range.copy()

        weights = (1.0 - selectivity) + selectivity * hand_strengths
        adjusted = opp_range * weights.astype(np.float32)
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
        else:
            return opp_range.copy()
        return adjusted

    def _confidence(self) -> float:
        """Observation confidence in [0, 1]."""
        return min(1.0, self._hands_observed / _CONFIDENCE_HANDS)
