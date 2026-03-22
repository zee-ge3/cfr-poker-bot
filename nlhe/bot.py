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
from nlhe.cfr.abstraction import ACTION_NAMES, PREFLOP_N_ACTIONS
import numpy as np

_PREFLOP_TABLE = None  # lazy load


def _get_preflop_table() -> PreflopTable:
    global _PREFLOP_TABLE
    if _PREFLOP_TABLE is None:
        _PREFLOP_TABLE = PreflopTable()
    return _PREFLOP_TABLE


class Bot:
    def __init__(self, budget_seconds: float = 5.0):
        self.budget_seconds = budget_seconds
        self._hole_cards: list[str] = []
        self._position: int = 0
        self._solver: NLHESolver | None = None

    def new_hand(self, hole_cards: list[str], position: int):
        """Reset state for a new hand."""
        self._hole_cards = hole_cards
        self._position = position
        self._solver = None

    def observe_action(self, action: str, amount: float = 0.0):
        """Update opponent range after they act. Call this before decide()."""
        if self._solver is not None:
            self._solver.observe_action(action)

    def decide(self, state: GameState) -> str:
        """Return the recommended action for the current game state."""
        if state.street == STREET_PREFLOP:
            return self._decide_preflop(state)
        return self._decide_postflop(state)

    def _decide_preflop(self, state: GameState) -> str:
        from nlhe.cfr.equity import card_str_to_idx
        try:
            table = _get_preflop_table()
            idxs = tuple(sorted(card_str_to_idx(c) for c in self._hole_cards))
            probs = table.lookup(idxs, self._position)
            # probs is shape (5,) over [FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN]
            # Map to valid_actions
            action_map = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN']
            valid = state.valid_actions
            # Filter to valid actions and renormalize
            weights = np.zeros(len(valid))
            for i, a in enumerate(valid):
                if a in action_map:
                    weights[i] = probs[action_map.index(a)]
                elif a == 'RAISE_OVERBET':
                    weights[i] = 0.0  # not in preflop table
            if weights.sum() < 1e-10:
                weights = np.ones(len(valid))
            weights /= weights.sum()
            chosen = np.random.choice(len(valid), p=weights)
            return valid[chosen]
        except Exception:
            # Fallback: call
            return 'CALL' if 'CALL' in state.valid_actions else state.valid_actions[0]

    def _decide_postflop(self, state: GameState) -> str:
        self._solver = NLHESolver(
            state, self._hole_cards, budget_seconds=self.budget_seconds
        )
        return self._solver.solve()
