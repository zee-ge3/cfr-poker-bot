"""
Abstract action enum for CFR solvers.
6 abstract actions at postflop decision nodes; preflop tables use 5 (no RAISE_OVERBET).
"""

# Abstract action indices
FOLD          = 0
CALL          = 1
RAISE_SMALL   = 2   # 0.33× pot opening, 2.2× facing
RAISE_LARGE   = 3   # 0.75× pot opening, 3× facing
RAISE_ALLIN   = 4   # actual all-in (remaining stack)
RAISE_OVERBET = 5   # 1.5× pot opening, pot-sized facing

N_ACTIONS = 6
PREFLOP_N_ACTIONS = 5  # preflop tables use indices 0-4 only

# Action names for logging
ACTION_NAMES = {
    FOLD: 'FOLD', CALL: 'CALL', RAISE_SMALL: 'RAISE_SMALL',
    RAISE_LARGE: 'RAISE_LARGE', RAISE_ALLIN: 'RAISE_ALLIN',
    RAISE_OVERBET: 'RAISE_OVERBET',
}

# Variance rank: lower = lower variance (used by variance_control.py)
VARIANCE_RANK = {
    FOLD: 0, CALL: 1, RAISE_SMALL: 2, RAISE_LARGE: 3,
    RAISE_OVERBET: 4, RAISE_ALLIN: 5,
}

# All raise actions (convenience tuple)
ALL_RAISES = (RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, RAISE_OVERBET)


def resolve_raise_amount(abstract_action: int, pot: float, stack: float,
                         current_bet: float = 0.0) -> float:
    """
    Map abstract raise action to concrete chip amount.
    Returns raise amount in chips (capped at stack).
    """
    if abstract_action == RAISE_ALLIN:
        return stack
    elif abstract_action == RAISE_OVERBET:
        if current_bet > 0:
            amount = pot * 1.0       # pot-sized reraise
        else:
            amount = pot * 1.5       # 1.5× pot overbet
    elif abstract_action == RAISE_SMALL:
        if current_bet > 0:
            amount = current_bet * 2.2
        else:
            amount = pot * 0.33
    elif abstract_action == RAISE_LARGE:
        if current_bet > 0:
            amount = current_bet * 3.0
        else:
            amount = pot * 0.75
    else:
        raise ValueError(f"Not a raise action: {abstract_action}")
    return min(amount, stack)


def abstract_action_from_real(action_str: str, raise_amount: float,
                               pot: float, stack: float) -> int:
    """
    Map a real game action to the nearest abstract action.
    """
    if action_str in ('FOLD',):
        return FOLD
    if action_str in ('CALL', 'CHECK'):
        return CALL
    if action_str == 'RAISE':
        if stack > 0 and raise_amount >= stack * 0.85:
            return RAISE_ALLIN
        # Use pre-raise pot as denominator (pot includes the raise already)
        pre_raise_pot = max(1.0, pot - raise_amount)
        frac = raise_amount / pre_raise_pot
        if frac <= 0.50:
            return RAISE_SMALL
        elif frac <= 0.90:
            return RAISE_LARGE
        elif frac <= 1.30:
            return RAISE_OVERBET
        else:
            return RAISE_ALLIN
    return CALL  # safe fallback
