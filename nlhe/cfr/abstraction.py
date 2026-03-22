"""
Action abstraction and hand indexing for NLHE CFR solvers.

Action constants mirror the Trips Poker competition code.
Card indexing: card_idx = rank * 4 + suit
  rank: 0=2, 1=3, ..., 12=A
  suit: 0=c, 1=d, 2=h, 3=s
"""

from itertools import combinations
import numpy as np

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

FOLD          = 0
CALL          = 1
RAISE_SMALL   = 2   # 0.33x pot opening
RAISE_LARGE   = 3   # 0.75x pot opening
RAISE_ALLIN   = 4   # actual all-in (remaining stack)
RAISE_OVERBET = 5   # 1.5x pot opening

N_ACTIONS         = 6
PREFLOP_N_ACTIONS = 5  # preflop tables use indices 0-4 only

ACTION_NAMES = {
    FOLD: 'FOLD',
    CALL: 'CALL',
    RAISE_SMALL: 'RAISE_SMALL',
    RAISE_LARGE: 'RAISE_LARGE',
    RAISE_ALLIN: 'RAISE_ALLIN',
    RAISE_OVERBET: 'RAISE_OVERBET',
}


def resolve_raise_amount(abstract_action: int, pot: float, stack: float,
                          current_bet: float = 0.0) -> float:
    """
    Map abstract raise action to concrete chip amount.
    Returns raise amount in chips, capped at stack.
    """
    if abstract_action == RAISE_ALLIN:
        return stack
    elif abstract_action == RAISE_OVERBET:
        if current_bet > 0:
            amount = pot * 1.0        # pot-sized reraise
        else:
            amount = pot * 1.5        # 1.5x pot overbet
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


# ---------------------------------------------------------------------------
# Hand indexing — 52-card NLHE, 1326 two-card combos
# ---------------------------------------------------------------------------

ALL_HANDS: list[tuple[int, int]] = list(combinations(range(52), 2))
HAND_TO_IDX: dict[tuple[int, int], int] = {h: i for i, h in enumerate(ALL_HANDS)}

# HAND_CONTAINS_CARD[card_idx][hand_idx] == True iff that hand contains card_idx
HAND_CONTAINS_CARD: np.ndarray = np.zeros((52, 1326), dtype=bool)
for _hand_idx, (_c1, _c2) in enumerate(ALL_HANDS):
    HAND_CONTAINS_CARD[_c1, _hand_idx] = True
    HAND_CONTAINS_CARD[_c2, _hand_idx] = True


# ---------------------------------------------------------------------------
# 169-bucket preflop abstraction
# ---------------------------------------------------------------------------

def _build_bucket_lookup() -> np.ndarray:
    """
    Return a (1326,) int32 array mapping each hand index to one of 169
    preflop bucket classes:
      0-12   : pairs  22..AA  (bucket = rank, so 22=0, AA=12)
      13-90  : suited hands, sorted by r_hi desc then r_lo desc
      91-168 : offsuit hands, same ordering
    """
    buckets = np.zeros(1326, dtype=np.int32)

    # Build ordered lists of (r_hi, r_lo) for suited and offsuit non-pairs
    suited_classes = []
    offsuit_classes = []
    for r_hi in range(12, -1, -1):       # 12=A down to 0=2
        for r_lo in range(r_hi - 1, -1, -1):   # strictly lower rank
            suited_classes.append((r_hi, r_lo))
            offsuit_classes.append((r_hi, r_lo))

    # suited_classes[0] = (12,11) = AK suited -> bucket 13
    # offsuit_classes[0] = (12,11) = AK offsuit -> bucket 91

    suited_bucket_map  = {cls: 13 + i for i, cls in enumerate(suited_classes)}
    offsuit_bucket_map = {cls: 91 + i for i, cls in enumerate(offsuit_classes)}

    for hand_idx, (c1, c2) in enumerate(ALL_HANDS):
        r1, s1 = c1 // 4, c1 % 4
        r2, s2 = c2 // 4, c2 % 4

        if r1 == r2:
            # Pocket pair: bucket = rank (0=22 ... 12=AA)
            buckets[hand_idx] = r1
        else:
            r_hi = max(r1, r2)
            r_lo = min(r1, r2)
            if s1 == s2:
                buckets[hand_idx] = suited_bucket_map[(r_hi, r_lo)]
            else:
                buckets[hand_idx] = offsuit_bucket_map[(r_hi, r_lo)]

    return buckets


HAND_BUCKET: np.ndarray = _build_bucket_lookup()
