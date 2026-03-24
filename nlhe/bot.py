# nlhe/bot.py
"""
Bot class for NLHE CFR-D play.

Three public methods:
    bot.new_hand(hole_cards, position)     # reset for a new hand
    bot.observe_action(action, amount)     # Bayesian opponent range update
    bot.decide(game_state) -> str          # run CFR-D, return action

No gym_env dependency.
"""
from nlhe.game import GameState, STREET_PREFLOP
from nlhe.cfr.solver import NLHESolver
from nlhe.cfr.preflop import PreflopTable
from nlhe.cfr.opponent import OpponentModel
from nlhe.cfr.abstraction import ACTION_NAMES, PREFLOP_N_ACTIONS
import numpy as np

_PREFLOP_TABLE = None  # lazy load


def _get_preflop_table() -> PreflopTable:
    global _PREFLOP_TABLE
    if _PREFLOP_TABLE is None:
        _PREFLOP_TABLE = PreflopTable()
    return _PREFLOP_TABLE


def _preflop_hand_tier(hole_cards: list[str]) -> int:
    """Classify hand into tiers for facing re-raises at deep stacks.

    Returns 1 (premium), 2 (strong), 3 (playable), 4 (fold to 3-bet).

    Tier 1: AA, KK, QQ, AKs  (~2.6% of combos)
    Tier 2: JJ, TT, 99, AKo, AQs, AQo, AJs, KQs (~6.5%)
    Tier 3: 88, 77, ATs, KJs, QJs, JTs, AJo, KJo, A9s, A5s (~8%)
    Tier 4: everything else
    """
    from nlhe.cfr.equity import card_str_to_idx
    idxs = [card_str_to_idx(c) for c in hole_cards]
    r1, r2 = idxs[0] // 4, idxs[1] // 4  # 0=2..12=A
    s1, s2 = idxs[0] % 4, idxs[1] % 4
    high, low = max(r1, r2), min(r1, r2)
    suited = (s1 == s2)
    pair = (r1 == r2)

    if pair:
        if high >= 10:   # QQ+
            return 1
        if high >= 7:    # 99-JJ
            return 2
        if high >= 5:    # 77-88
            return 3
        return 4

    if high == 12:  # Ace-high
        if low >= 11:            # AK
            return 1 if suited else 2
        if low >= 10:            # AQ
            return 2
        if low == 9:             # AJ
            return 2 if suited else 3
        if low == 8 and suited:  # ATs
            return 3
        if low == 7 and suited:  # A9s
            return 3
        if low == 3 and suited:  # A5s
            return 3
        return 4

    if high == 11: # King-high
        if low == 10 and suited:  # KQs
            return 2
        if low >= 9 and suited:   # KJs
            return 3
        if low == 10:             # KJo
            return 3
    
    if high == 10 and low == 9 and suited: # QJs
        return 3
    
    if high == 9 and low == 8 and suited: # JTs
        return 3

    return 4


# Time budget split per street
_STREET_BUDGET = {1: 0.35, 2: 0.30, 3: 0.35}


class Bot:
    def __init__(self, budget_seconds: float = 5.0):
        self.budget_seconds = budget_seconds
        self._hole_cards: list[str] = []
        self._position: int = 0
        self._solver: NLHESolver | None = None
        self._opponent = OpponentModel()
        self._last_street: int = -1
        self._saved_opp_range: np.ndarray | None = None

    def new_hand(self, hole_cards: list[str], position: int):
        """Reset state for a new hand."""
        self._hole_cards = hole_cards
        self._position = position
        self._solver = None
        self._last_street = -1
        self._opponent.new_hand()
        self._saved_opp_range = None
        self._preflop_raises_seen = 0

    def observe_action(self, action: str, amount: float = 0.0):
        """Update opponent range and model after they act."""
        self._opponent.observe_action(action, street=max(self._last_street, 0))
        if action.startswith("RAISE"):
            self._opponent.observe_response_to_our_raise(action)
        if self._solver is not None:
            self._solver.observe_action(action)
            # Save solver-narrowed range for cross-street tracking
            self._saved_opp_range = self._solver.opp_range.copy()
        elif action.startswith("RAISE") and self._last_street <= 0:
            # Preflop raise without solver
            self._preflop_raises_seen += 1

    def decide(self, state: GameState) -> str:
        """Return the recommended action for the current game state.
        
        Uses the real-time CFR-D solver for all streets, which is now
        mathematically exact for preflop due to the mapped 1326x1326
        equity matrix, and protects against depth-limit artifacts
        via SPR-based action abstraction.
        """
        return self._decide_postflop(state)

    def _decide_preflop(self, state: GameState) -> str:
        from nlhe.cfr.preflop import PreflopTable
        from nlhe.cfr.equity import card_str_to_idx
        self._last_street = 0
        try:
            # We use a singleton or cached table ideally, but this is fast enough
            if not hasattr(self, '_preflop_table'):
                self._preflop_table = PreflopTable()
            
            idxs = tuple(sorted(card_str_to_idx(c) for c in self._hole_cards))
            probs = np.array(self._preflop_table.lookup(idxs, self._position), dtype=float)
            
            # The table outputs probabilities for [FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN]
            # At deep stacks (SPR > 5), redirect RAISE_ALLIN probability to RAISE_LARGE
            effective = min(state.our_stack, state.opp_stack)
            spr = effective / max(state.pot, 1)
            if spr > 5:
                probs[3] += probs[4]
                probs[4] = 0.0
                
            action_map = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN']
            valid = state.valid_actions
            weights = np.zeros(len(valid))
            
            for i, a in enumerate(valid):
                if a in action_map:
                    weights[i] = probs[action_map.index(a)]
                elif a == 'RAISE_OVERBET':
                    # Allow RAISE_OVERBET to absorb RAISE_ALLIN probability if large was 0
                    if spr > 5 and 'RAISE_LARGE' not in valid:
                         weights[i] = probs[4]
                    else:
                         weights[i] = 0.0
                    
            if weights.sum() < 1e-10:
                weights = np.ones(len(valid))
                
            weights /= weights.sum()
            chosen = np.random.choice(len(valid), p=weights)
            return valid[chosen]
        except Exception as e:
            print("Preflop table error:", e)
            return 'CALL' if 'CALL' in state.valid_actions else state.valid_actions[0]

    def _decide_postflop(self, state: GameState) -> str:
        street = state.street
        budget = self.budget_seconds * _STREET_BUDGET.get(street, 0.33)

        # Standard Action Abstraction for deep stacks:
        # Depth-limited solvers over-value ALL-IN and massive overbets on early streets
        # because the leaf evaluation assumes no further betting (check-down).
        # We restrict the maximum bet size based on SPR (Stack-to-Pot Ratio)
        # to ensure the tree remains realistic. River (street=3) is solved exactly.
        filtered_actions = list(state.valid_actions)
        effective_stack = min(state.our_stack, state.opp_stack)
        spr = effective_stack / max(state.pot, 1)

        if street < 3:
            if spr > 2.0:
                if 'RAISE_ALLIN' in filtered_actions:
                    filtered_actions.remove('RAISE_ALLIN')
            if spr > 4.0:
                if 'RAISE_OVERBET' in filtered_actions:
                    filtered_actions.remove('RAISE_OVERBET')

        # If we filtered everything except Fold/Call, restore the largest available raise
        if not any(a.startswith('RAISE') for a in filtered_actions) and any(a.startswith('RAISE') for a in state.valid_actions):
            for a in ['RAISE_LARGE', 'RAISE_SMALL', 'RAISE_OVERBET', 'RAISE_ALLIN']:
                if a in state.valid_actions:
                    filtered_actions.append(a)
                    break

        # Create a modified state with the filtered actions for the solver
        import copy
        solver_state = copy.copy(state)
        solver_state.valid_actions = filtered_actions

        from nlhe.cfr.solver import NLHESolver
        # Always create fresh solver
        self._solver = NLHESolver(
            solver_state, self._hole_cards,
            budget_seconds=budget,
            opponent_model=self._opponent,
            initial_opp_range=self._saved_opp_range,
        )
        self._last_street = street
        return self._solver.solve()
