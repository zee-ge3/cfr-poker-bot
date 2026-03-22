"""
Tournament utility function for CFR solvers.
Maps chip outcomes to Match Win Probability (MWP) change.

scipy is NOT available — uses math.erf for the normal CDF.
"""
import math
import numpy as np


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (scipy not available)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def tournament_utility(chips_won: float,
                       match_margin: float,
                       hands_remaining: int,
                       sigma_per_hand: float = 20.0) -> float:
    """
    Map chip outcome to change in Match Win Probability (delta MWP).

    Optimizing delta MWP instead of raw chip EV makes CFR organically:
    - Risk-averse when ahead (concave region of CDF)
    - Risk-seeking when behind (convex region of CDF)

    Args:
        chips_won: chips won/lost this hand (negative if lost)
        match_margin: current cumulative chip lead (positive = ahead)
        hands_remaining: estimated hands left in match (used for sigma scaling)
        sigma_per_hand: per-hand chip std dev; default 20 calibrated from data

    Returns:
        delta MWP: change in probability of winning the match (float in (-1, 1))
    """
    sigma_total = sigma_per_hand * math.sqrt(max(1, hands_remaining))
    return (norm_cdf((match_margin + chips_won) / sigma_total)
            - norm_cdf(match_margin / sigma_total))


def tournament_utility_vec(chips_won: np.ndarray,
                           match_margin: float,
                           hands_remaining: int,
                           sigma_per_hand: float = 20.0) -> np.ndarray:
    """Vectorized version of tournament_utility."""
    sigma_total = sigma_per_hand * math.sqrt(max(1, hands_remaining))
    
    def norm_cdf_vec(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        
    return (norm_cdf_vec((match_margin + chips_won) / sigma_total)
            - norm_cdf_vec(match_margin / sigma_total)).astype(np.float32)
