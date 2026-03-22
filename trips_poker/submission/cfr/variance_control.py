"""
Variance-Biased Equilibrium Selection for tournament poker.

Fix #26: CFV-weighted action selection replaces blind veto + avg_strategy sampling.
Old: binary veto (0.005 MWP threshold) blocked ZERO shoves across all matches
because CFV spread is always < 0.005. Then avg_strategy (scalar, hand-blind)
decided everything — weak hands shoved at the same rate as nut hands.

New: use CFVs to reweight avg_strategy directly. Actions with higher CFV get
proportionally more probability. This preserves equilibrium mixing while
steering away from -EV actions for specific hands.

Fix #24: Protection/pressure modes BIAS the sampling distribution
toward low/high variance rather than deterministically picking the extreme.
"""
import random
import numpy as np
from submission.cfr.action_abstraction import VARIANCE_RANK, N_ACTIONS, ALL_RAISES

MIN_ITERATIONS_FOR_TIEBREAK = 40  # below this, average strategy is unreliable

# How strongly CFVs reweight avg_strategy. Higher = more CFV influence.
# At 0: pure avg_strategy. At very high values: approaches CFV argmax.
CFV_REWEIGHT_TEMPERATURE = 500.0

# How strongly protection/pressure modes bias sampling.
VARIANCE_BIAS_STRENGTH = 0.6


def select_action(cfvs: np.ndarray, valid_actions: list, mode: str,
                  iterations_run: int = 999,
                  avg_strategy: np.ndarray = None) -> int:
    """
    Select action using CFV-reweighted avg_strategy.

    Args:
        cfvs: shape (N_ACTIONS,) float32 — hand-specific CFVs from pass-2
        valid_actions: list of valid abstract action indices
        mode: 'neutral' | 'protection' | 'pressure'
        iterations_run: how many CFR iterations have completed
        avg_strategy: shape (N_ACTIONS,) float32 — average strategy from CFR root

    Returns:
        Selected abstract action index (int).
    """
    if not valid_actions:
        return 1  # CALL as safe default

    # Too few iterations — use CALL as safe default (CFR output unreliable)
    if iterations_run < MIN_ITERATIONS_FOR_TIEBREAK:
        if avg_strategy is not None:
            return max(valid_actions, key=lambda a: float(avg_strategy[a]))
        return 1  # CALL

    has_cfvs = np.any(cfvs != 0)

    if has_cfvs and avg_strategy is not None:
        return _cfv_weighted_selection(cfvs, avg_strategy, valid_actions, mode)

    # avg_strategy only (no CFVs)
    if avg_strategy is not None:
        return _select_from_avg_strategy(avg_strategy, valid_actions, mode)

    # Pure CFV fallback (no avg_strategy available)
    return max(valid_actions, key=lambda a: float(cfvs[a]))


def _cfv_weighted_selection(cfvs, avg_strategy, valid_actions, mode):
    """Reweight avg_strategy by CFV signal, then sample.

    Instead of binary veto, we multiply each action's avg_strategy probability
    by a weight derived from its CFV relative to the best CFV. Actions close
    to best_cfv keep their probability; actions far below get suppressed.
    """
    cfv_vals = {a: float(cfvs[a]) for a in valid_actions}
    best_cfv = max(cfv_vals.values())

    # Compute CFV-based weights: exp(temperature * (cfv - best_cfv))
    # This gives 1.0 for the best action and exponentially decays for worse ones
    weights = {}
    for a in valid_actions:
        delta = cfv_vals[a] - best_cfv  # always <= 0
        weights[a] = np.exp(CFV_REWEIGHT_TEMPERATURE * delta)

    # Multiply avg_strategy by CFV weights
    combined = {}
    for a in valid_actions:
        combined[a] = float(avg_strategy[a]) * weights[a]

    # If all combined probs are zero (avg_strategy was zero for all viable),
    # fall back to pure CFV weights
    total = sum(combined.values())
    if total <= 1e-30:
        combined = weights
        total = sum(combined.values())

    if total <= 1e-30:
        return valid_actions[0]

    # Normalize
    combined = {a: v / total for a, v in combined.items()}

    # Apply variance bias for protection/pressure modes
    if mode in ('protection', 'pressure'):
        combined = _apply_variance_bias(combined, mode)

    return _sample_from_probs(combined, valid_actions)


def _sample_from_probs(probs: dict, valid_actions: list) -> int:
    """Sample a single action from a probability dict."""
    candidates = [a for a in valid_actions if probs.get(a, 0) > 0]
    if not candidates:
        candidates = valid_actions

    weights = [probs.get(a, 0) for a in candidates]
    total = sum(weights)
    if total <= 0:
        return random.choice(candidates)

    r = random.random() * total
    cumulative = 0.0
    for a, w in zip(candidates, weights):
        cumulative += w
        if r < cumulative:
            return a
    return candidates[-1]


def _select_from_avg_strategy(avg_strategy: np.ndarray, valid_actions: list,
                              mode: str) -> int:
    """Select action from CFR average strategy with variance bias.

    All modes sample from the strategy distribution. Protection/pressure
    modes reweight probabilities to favor low/high variance actions
    while preserving the equilibrium's relative ordering.
    """
    probs = {a: float(avg_strategy[a]) for a in valid_actions}
    max_prob = max(probs.values())

    if max_prob <= 0:
        # All avg_strategy mass is on actions not in valid_actions.
        # NEVER randomly fold — default to CALL.
        if 1 in valid_actions:  # CALL
            return 1
        return valid_actions[0]

    # Apply variance bias for protection/pressure modes
    if mode in ('protection', 'pressure'):
        probs = _apply_variance_bias(probs, mode)

    # Sample from the (possibly biased) distribution
    candidates = [a for a in valid_actions if probs[a] > 0]
    if not candidates:
        candidates = valid_actions

    weights = [probs.get(a, 0) for a in candidates]
    total = sum(weights)
    if total <= 0:
        return random.choice(candidates)

    r = random.random() * total
    cumulative = 0.0
    for a, w in zip(candidates, weights):
        cumulative += w
        if r < cumulative:
            return a
    return candidates[-1]


def _apply_variance_bias(probs: dict, mode: str) -> dict:
    """Reweight action probabilities by variance rank.

    Protection mode: multiply each action's probability by a factor that
    decreases with variance rank (low variance favored).
    Pressure mode: multiply by a factor that increases with variance rank.

    The bias is proportional to VARIANCE_BIAS_STRENGTH. At strength=0,
    no change. At strength=1, extreme bias (but still probabilistic).
    """
    if not probs:
        return probs

    s = VARIANCE_BIAS_STRENGTH
    max_vr = max(VARIANCE_RANK.values())  # 5 (ALL-IN)

    biased = {}
    for a, p in probs.items():
        if p <= 0:
            biased[a] = 0.0
            continue

        vr = VARIANCE_RANK.get(a, 2)
        # Normalize variance rank to [0, 1]
        norm_vr = vr / max(1, max_vr)

        if mode == 'protection':
            # Low variance favored: multiply by (1 - s * norm_vr)
            # FOLD (vr=0) gets factor 1.0, ALL-IN (vr=1) gets factor (1-s)
            factor = 1.0 - s * norm_vr
        else:  # pressure
            # High variance favored: multiply by (1 - s * (1 - norm_vr))
            # ALL-IN (vr=1) gets factor 1.0, FOLD (vr=0) gets factor (1-s)
            factor = 1.0 - s * (1.0 - norm_vr)

        biased[a] = p * max(0.01, factor)  # floor at 1% to never fully zero out

    # Normalize
    total = sum(biased.values())
    if total > 0:
        biased = {a: v / total for a, v in biased.items()}

    return biased
