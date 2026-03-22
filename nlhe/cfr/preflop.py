"""
Load and query pre-solved NLHE preflop CFR+ tables.
Tables are computed offline by preflop_compute.py and committed to the repo.
"""
import os
import numpy as np
from nlhe.cfr.abstraction import HAND_BUCKET, HAND_TO_IDX, PREFLOP_N_ACTIONS

TABLE_PATH = os.path.join(os.path.dirname(__file__), 'tables', 'preflop_strategy.npy')


class PreflopTable:
    def __init__(self, path: str = TABLE_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preflop table not found at {path}. "
                "Run: python -m nlhe.cfr.preflop_compute"
            )
        self.strategy = np.load(path)  # shape (169, 2, 5)
        assert self.strategy.shape == (169, 2, PREFLOP_N_ACTIONS), \
            f"Unexpected shape: {self.strategy.shape}"

    def lookup(self, our_hole_idxs: tuple, position: int) -> np.ndarray:
        """
        Look up preflop action probabilities for a given hand and position.

        our_hole_idxs: tuple of two 0-51 card indices
        position: 0=SB, 1=BB

        Returns: np.ndarray of shape (5,) with action probabilities.
        """
        h = tuple(sorted(our_hole_idxs))
        if h not in HAND_TO_IDX:
            raise ValueError(f"Invalid hand indices: {h}")
        hand_idx = HAND_TO_IDX[h]
        bucket = HAND_BUCKET[hand_idx]
        return self.strategy[bucket, position, :]
