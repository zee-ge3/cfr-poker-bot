"""
Variance control — CFV-weighted action selection.

Ported from trips_poker competition code. Replaces binary CFV veto with
proportional reweighting based on counterfactual values.
"""
from __future__ import annotations

import numpy as np

# Minimum iterations before CFV reweighting kicks in.
MIN_ITERATIONS = 40

# Temperature for CFV->weight conversion.
# Calibrated for chip EV scale (pot ~2-200 chips).
# Competition used 500.0 for MWP scale (~0.02); chip EV needs ~2.0.
CFV_TEMPERATURE = 2.0


def select_action(
    avg_strategy: np.ndarray,
    cfvs: np.ndarray,
    iterations: int,
) -> int:
    """
    Select an action index using CFV-weighted average strategy.

    Below MIN_ITERATIONS: sample from raw avg_strategy.
    Above MIN_ITERATIONS: weight avg_strategy by exp(temperature * (cfv - best)),
    renormalize, then sample.

    Parameters
    ----------
    avg_strategy : (n_actions,) average strategy from CFR
    cfvs : (n_actions,) counterfactual values per action
    iterations : number of CFR iterations completed

    Returns
    -------
    int — selected action index
    """
    n = len(avg_strategy)
    if n == 0:
        raise ValueError("Empty strategy")
    if n == 1:
        return 0

    if iterations < MIN_ITERATIONS:
        probs = avg_strategy.astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n, dtype=np.float64) / n
        return int(np.random.choice(n, p=probs))

    best_cfv = cfvs.max()
    weights = np.exp(CFV_TEMPERATURE * (cfvs - best_cfv))

    blended = (avg_strategy * weights).astype(np.float64)
    total = blended.sum()
    if total > 0:
        blended /= total
    else:
        blended = np.ones(n, dtype=np.float64) / n

    return int(np.random.choice(n, p=blended))
