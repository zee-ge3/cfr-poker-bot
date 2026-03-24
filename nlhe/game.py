"""
NLHE Game Engine for the portfolio Streamlit demo.

Supports a two-player heads-up No-Limit Hold'em hand. Designed to be
independent of the competition gym_env — it uses treys for card dealing and
evaluation, and nlhe.cfr.abstraction for raise sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import treys

from nlhe.cfr.abstraction import (
    resolve_raise_amount,
    RAISE_SMALL,
    RAISE_LARGE,
    RAISE_ALLIN,
    RAISE_OVERBET,
    FOLD,
    CALL,
)

# ---------------------------------------------------------------------------
# Street constants
# ---------------------------------------------------------------------------

STREET_PREFLOP = 0
STREET_FLOP    = 1
STREET_TURN    = 2
STREET_RIVER   = 3

STREET_NAMES = {
    STREET_PREFLOP: 'preflop',
    STREET_FLOP:    'flop',
    STREET_TURN:    'turn',
    STREET_RIVER:   'river',
}

# Number of board cards dealt at the start of each street
_BOARD_CARDS = {
    STREET_PREFLOP: 0,
    STREET_FLOP:    3,
    STREET_TURN:    1,
    STREET_RIVER:   1,
}

# ---------------------------------------------------------------------------
# GameState dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    street: int           # 0=preflop, 1=flop, 2=turn, 3=river
    pot: int              # chips in pot
    our_stack: int
    opp_stack: int
    our_hole: list        # e.g. ['Ah', 'Kd']
    opp_hole: list        # empty during play; populated at showdown reveal
    board: list           # 0–5 treys card strings
    valid_actions: list   # subset of action-name strings
    position: int         # 0=BTN/SB acts first preflop, 1=BB
    street_bets: list     # ordered sequence of individual bet amounts this street
    to_call: int = 0      # outstanding bet hero must match (0 if no bet)


# ---------------------------------------------------------------------------
# NLHEGame
# ---------------------------------------------------------------------------

class NLHEGame:
    """
    Two-player heads-up NLHE engine.

    Positions
    ---------
    * position 0 — BTN / SB — posts 1 chip, acts first preflop
    * position 1 — BB       — posts 2 chips, acts second preflop

    Post-flop: BB (position 1) acts first on every subsequent street.

    Action strings
    --------------
    'FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN', 'RAISE_OVERBET'

    Return value of human_action / bot_action
    -----------------------------------------
    When hand continues::

        {'hand_over': False, 'street': int, 'state': GameState}

    When hand is over::

        {'hand_over': True, 'winner': str, 'winnings': int,
         'bot_hole': list, 'reason': str}
    """

    def __init__(self, starting_stack: int = 100) -> None:
        self.starting_stack = starting_stack
        self._evaluator = treys.Evaluator()

        # These are set / reset by new_hand
        self._deck: Optional[treys.Deck] = None
        self._human_pos: int = 0       # 0=SB, 1=BB

        # Hole cards as treys ints (internal)
        self._human_hole_ints: list = []
        self._bot_hole_ints: list   = []

        # Board cards as treys ints (internal)
        self._board_ints: list = []

        # Stacks and pot
        self._human_stack: int = 0
        self._bot_stack: int   = 0
        self._pot: int         = 0

        # Betting state
        self._street: int           = STREET_PREFLOP
        self._current_bet: int      = 0   # amount the next actor must call
        self._street_bets: list     = []  # running list of bets this street
        self._actions_this_street: int = 0
        # Whose turn it is: 0=human, 1=bot
        self._actor: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_hand(self, human_position: int = 0) -> GameState:
        """
        Deal a new hand. Post blinds (SB=1, BB=2). Return initial GameState.
        """
        self._human_pos = human_position  # 0=SB, 1=BB

        # Fresh deck and stacks
        self._deck = treys.Deck()
        self._human_stack = self.starting_stack
        self._bot_stack   = self.starting_stack
        self._pot         = 0

        # Deal hole cards (treys ints)
        cards = self._deck.draw(4)
        self._human_hole_ints = cards[:2]
        self._bot_hole_ints   = cards[2:]

        self._board_ints = []
        self._street     = STREET_PREFLOP

        # Post blinds —————————————————————————————————————————————
        # SB posts 1, BB posts 2
        if self._human_pos == 0:
            # Human is SB
            sb_post = min(1, self._human_stack)
            bb_post = min(2, self._bot_stack)
            self._human_stack -= sb_post
            self._bot_stack   -= bb_post
        else:
            # Human is BB
            sb_post = min(1, self._bot_stack)
            bb_post = min(2, self._human_stack)
            self._bot_stack   -= sb_post
            self._human_stack -= bb_post

        self._pot = sb_post + bb_post

        # After blinds: SB put in 1, BB put in 2.  SB acts first preflop.
        # SB must call 1 more to match BB's 2.
        # _street_contrib[i] tracks how much player i has invested this street.
        # _to_call is always the net additional chips the current actor must add.
        self._to_call: int = 1   # SB must call 1 more
        self._street_bets  = [sb_post, bb_post]
        self._actions_this_street = 0

        # Per-player street contributions (index 0=human, 1=bot).
        # Used to compute _to_call correctly after raises.
        if self._human_pos == 0:
            # Human is SB: contributed sb_post; bot is BB: contributed bb_post
            self._street_contrib: list = [sb_post, bb_post]
        else:
            # Human is BB: contributed bb_post; bot is SB: contributed sb_post
            self._street_contrib: list = [bb_post, sb_post]

        # Preflop: SB / BTN (position 0) acts first
        # After blinds, actors alternate starting with position 0 (SB)
        if self._human_pos == 0:
            # Human is SB → human acts first preflop
            self._actor = 0  # 0=human
        else:
            # Human is BB → bot (SB) acts first preflop
            self._actor = 1  # 1=bot

        return self._get_state()

    def human_action(self, action: str, raise_amount: int = 0) -> dict:
        """Process a human action. Returns result dict."""
        if self._actor != 0:
            raise RuntimeError("Not human's turn")
        return self._process_action(actor=0, action=action, raise_amount=raise_amount)

    def bot_action(self, action: str, raise_amount: int = 0) -> dict:
        """Process a bot action. Returns result dict."""
        if self._actor != 1:
            raise RuntimeError("Not bot's turn")
        return self._process_action(actor=1, action=action, raise_amount=raise_amount)

    def _force_street(self, street: int) -> None:
        """
        Test helper — forcibly advance to the given street, dealing board cards
        and resetting betting state. Does not check who acts; caller manages
        subsequent actor assignment.
        """
        while self._street < street:
            self._street += 1
            n = _BOARD_CARDS[self._street]
            self._board_ints.extend(self._deck.draw(n))

        # Reset betting for new street
        self._to_call = 0
        self._street_bets = []
        self._actions_this_street = 0
        self._street_contrib = [0, 0]

        # Post-flop: BB acts first (position 1)
        # After _force_street the next caller decides who acts;
        # default to post-flop order: BB acts first.
        if self._human_pos == 1:
            # Human is BB → human acts first post-flop
            self._actor = 0
        else:
            # Human is SB → bot (BB) acts first post-flop
            self._actor = 1

    def _get_state(self) -> GameState:
        """Build and return a GameState snapshot."""
        human_stack = self._human_stack
        bot_stack   = self._bot_stack

        our_hole  = [treys.Card.int_to_str(c) for c in self._human_hole_ints]
        board     = [treys.Card.int_to_str(c) for c in self._board_ints]

        valid = self._compute_valid_actions()

        return GameState(
            street       = self._street,
            pot          = self._pot,
            our_stack    = human_stack,
            opp_stack    = bot_stack,
            our_hole     = our_hole,
            opp_hole     = [],          # hidden until showdown
            board        = board,
            valid_actions= valid,
            position     = self._human_pos,
            street_bets  = list(self._street_bets),
            to_call      = self._to_call,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_valid_actions(self) -> list:
        """
        Compute valid action strings for the current actor.

        Rules:
        - FOLD only when there is a bet to call (_to_call > 0)
        - CALL always (covers both CHECK and CALL)
        - Raise actions always if actor has chips; RAISE_ALLIN when both players
          have chips, others when stack permits meaningful raise
        """
        actions = []

        if self._to_call > 0:
            actions.append('FOLD')

        actions.append('CALL')

        # Determine which player's stack matters for raises
        actor_stack = self._human_stack if self._actor == 0 else self._bot_stack

        if actor_stack > 0:
            actions.append('RAISE_ALLIN')

            # Only add sized raises if they would be meaningfully different from
            # an all-in and we have enough stack
            if actor_stack > self._to_call:
                # Compute raise sizes relative to current pot
                small  = int(resolve_raise_amount(RAISE_SMALL,   self._pot, actor_stack, self._to_call))
                large  = int(resolve_raise_amount(RAISE_LARGE,   self._pot, actor_stack, self._to_call))
                overbet= int(resolve_raise_amount(RAISE_OVERBET, self._pot, actor_stack, self._to_call))
                allin  = actor_stack

                # Add each only if strictly less than all-in
                if small < allin:
                    actions.append('RAISE_SMALL')
                if large < allin and large != small:
                    actions.append('RAISE_LARGE')
                if overbet < allin and overbet not in (small, large):
                    actions.append('RAISE_OVERBET')

        return actions

    def _process_action(self, actor: int, action: str, raise_amount: int = 0) -> dict:
        """
        Apply action, update state, and return result dict.

        actor: 0=human, 1=bot
        """
        if action == 'FOLD' and self._to_call == 0:
            # Treat FOLD as CHECK when no bet outstanding
            action = 'CALL'

        if action == 'FOLD':
            # Actor folds — opponent wins pot
            winner_actor = 1 - actor
            return self._end_hand(winner_actor, reason='fold')

        actor_stack_attr  = '_human_stack' if actor == 0 else '_bot_stack'
        actor_stack       = getattr(self, actor_stack_attr)

        if action == 'CALL':
            # Call or check
            chips = min(self._to_call, actor_stack)
            setattr(self, actor_stack_attr, actor_stack - chips)
            self._pot       += chips
            self._street_bets.append(chips)
            self._street_contrib[actor] += chips
            self._to_call    = 0
            self._actions_this_street += 1

            # Did a call (not a check) end the betting round?
            # A call ends the street when chips > 0 (there was a bet to call)
            # A check (chips == 0) needs both players to have checked.
            if chips > 0:
                # Preflop special case: SB limping doesn't end the street.
                # BB still gets the option to check or raise.
                if self._street == STREET_PREFLOP and self._actions_this_street < 2:
                    self._actor = 1 - actor
                    return {'hand_over': False, 'street': self._street, 'state': self._get_state()}
                # Otherwise a call ends the betting round
                return self._advance_street_or_showdown()
            else:
                # This was a CHECK; need opponent to also check
                if self._actions_this_street >= 2:
                    # Both have acted with no bet (double check)
                    return self._advance_street_or_showdown()
                else:
                    # Switch actor, continue
                    self._actor = 1 - actor
                    return {'hand_over': False, 'street': self._street, 'state': self._get_state()}

        elif action in ('RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN', 'RAISE_OVERBET'):
            action_const = {
                'RAISE_SMALL':   RAISE_SMALL,
                'RAISE_LARGE':   RAISE_LARGE,
                'RAISE_ALLIN':   RAISE_ALLIN,
                'RAISE_OVERBET': RAISE_OVERBET,
            }[action]

            if raise_amount > 0:
                chips = min(raise_amount, actor_stack)
            else:
                chips = int(resolve_raise_amount(action_const, self._pot, actor_stack, self._to_call))
                chips = max(chips, 1)  # at least 1 chip
                chips = min(chips, actor_stack)

            setattr(self, actor_stack_attr, actor_stack - chips)
            self._pot        += chips
            self._street_bets.append(chips)
            self._street_contrib[actor] += chips
            self._actions_this_street += 1

            # _to_call for the opponent is the net additional chips they must add
            # to match the raiser's total street investment.
            opponent = 1 - actor
            self._to_call = self._street_contrib[actor] - self._street_contrib[opponent]

            # Switch to opponent
            self._actor = opponent
            return {'hand_over': False, 'street': self._street, 'state': self._get_state()}

        else:
            raise ValueError(f"Unknown action: {action!r}")

    def _advance_street_or_showdown(self) -> dict:
        """
        Move to the next street (dealing board cards) or, if we're on river,
        go to showdown.
        """
        if self._street == STREET_RIVER:
            return self._showdown()

        # Advance to next street
        self._street += 1
        n = _BOARD_CARDS[self._street]
        self._board_ints.extend(self._deck.draw(n))

        # Reset betting
        self._to_call             = 0
        self._street_bets         = []
        self._actions_this_street = 0
        self._street_contrib      = [0, 0]

        # Post-flop: BB acts first (position 1 relative to table)
        # Human is BB when human_pos == 1 → human acts first post-flop
        if self._human_pos == 1:
            self._actor = 0   # human is BB, acts first
        else:
            self._actor = 1   # bot is BB, acts first

        return {'hand_over': False, 'street': self._street, 'state': self._get_state()}

    def _showdown(self) -> dict:
        """Evaluate hands and return showdown result."""
        board = self._board_ints

        human_score = self._evaluator.evaluate(board, self._human_hole_ints)
        bot_score   = self._evaluator.evaluate(board, self._bot_hole_ints)

        # Lower score = better hand in treys
        if human_score < bot_score:
            winner_actor = 0
        elif bot_score < human_score:
            winner_actor = 1
        else:
            # Tie — split pot
            split = self._pot // 2
            remainder = self._pot - split * 2
            self._human_stack += split + remainder
            self._bot_stack   += split
            self._pot          = 0
            bot_hole = [treys.Card.int_to_str(c) for c in self._bot_hole_ints]
            return {
                'hand_over': True,
                'winner':    'tie',
                'winnings':  split,
                'bot_hole':  bot_hole,
                'reason':    'showdown_tie',
            }

        return self._end_hand(winner_actor, reason='showdown')

    def _end_hand(self, winner_actor: int, reason: str) -> dict:
        """
        Award pot to winner_actor (0=human, 1=bot), return result dict.
        """
        winnings = self._pot

        if winner_actor == 0:
            self._human_stack += winnings
            winner_name = 'human'
        else:
            self._bot_stack   += winnings
            winner_name = 'bot'

        self._pot = 0

        bot_hole = [treys.Card.int_to_str(c) for c in self._bot_hole_ints]

        return {
            'hand_over': True,
            'winner':    winner_name,
            'winnings':  winnings,
            'bot_hole':  bot_hole,
            'reason':    reason,
        }
