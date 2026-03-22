"""
Opponent model: observes all opponent actions, exposes clean context dict.
Never makes decisions — only observes and reports.
"""
import math
from collections import deque
from submission.card_utils import rank, suit, is_suited, is_paired, classify_hand


class OpponentModel:
    def __init__(self):
        # Per-street action counts (street 0=preflop, 1=flop, 2=turn, 3=river)
        self._raises = [0, 0, 0, 0]
        self._calls = [0, 0, 0, 0]
        self._checks = [0, 0, 0, 0]
        self._folds = [0, 0, 0, 0]
        self._actions_total = [0, 0, 0, 0]

        # Fold response to OUR raises
        self._folds_to_our_raise = [0, 0, 0, 0]
        self._calls_to_our_raise = [0, 0, 0, 0]

        # Bet sizing history
        self._raise_sizes = deque(maxlen=200)

        # Bluff rate (Bayesian Beta distribution)
        self._bluff_alpha = 2.0
        self._bluff_beta = 8.0

        # Per-hand action sequence
        self._hand_actions = []
        self._raised_this_hand = False

        # Range preference weights
        self.pref_suited = 1.0
        self.pref_paired = 1.0

        # Showdown hand type tracking
        self._sd_hand_types = {}
        self._sd_count = 0

        # Discard pattern tracking
        self._discard_data = {
            'kept_suited_count': 0,
            'kept_pair_count': 0,
            'discarded_pair_count': 0,
            'total_observed': 0,
        }

        # Hand observation count (for confidence blending in v4 solver)
        self._hands_observed = 0

    def update_action(self, action: str, street: int, bet_size: int = 0, pot: int = 0):
        s = min(street, 3)
        self._actions_total[s] += 1

        if 'RAISE' in action:
            self._raises[s] += 1
            self._raised_this_hand = True
            if pot > 0 and bet_size > 0:
                self._raise_sizes.append((bet_size, pot))
            self._hand_actions.append('RAISE')
        elif 'CALL' in action:
            self._calls[s] += 1
            self._hand_actions.append('CALL')
        elif 'CHECK' in action:
            self._checks[s] += 1
            self._hand_actions.append('CHECK')
        elif 'FOLD' in action:
            self._folds[s] += 1
            self._hand_actions.append('FOLD')

    def update_fold_to_our_raise(self, street: int):
        s = min(street, 3)
        self._folds_to_our_raise[s] += 1

    def update_call_to_our_raise(self, street: int):
        s = min(street, 3)
        self._calls_to_our_raise[s] += 1

    def update_showdown(self, raised: bool = False, hand_strong: bool = False,
                        hand_score: int = None, raised_streets: list = None):
        if raised:
            # Faster learning rate: increment by 2.0 instead of 1.0
            if hand_strong:
                self._bluff_beta += 2.0
            else:
                self._bluff_alpha += 2.0

        if hand_score is not None:
            hand_type = classify_hand(hand_score)
            self._sd_hand_types[hand_type] = self._sd_hand_types.get(hand_type, 0) + 1
            self._sd_count += 1

    def update_discards(self, opp_discards):
        if len(opp_discards) != 3 or any(c == -1 for c in opp_discards):
            return

        self._discard_data['total_observed'] += 1
        ranks = [rank(c) for c in opp_discards]
        suits = [suit(c) for c in opp_discards]

        from collections import Counter
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Learning Rate for preference updates (V7 speedup)
        lr = 0.15

        # If they discard a pair, they likely prefer draws (suited/connected)
        if max(rank_counts.values()) >= 2:
            self._discard_data['discarded_pair_count'] += 1
            self.pref_paired = self.pref_paired * (1 - lr) + 0.5 * lr
            self.pref_suited = self.pref_suited * (1 - lr) + 1.8 * lr
        
        # If they keep suited cards (discarded all 3 suits), increase suited pref
        if len(suit_counts) == 3:
            self._discard_data['kept_suited_count'] += 1
            self.pref_suited = self.pref_suited * (1 - lr) + 2.0 * lr

        # If they keep a pair (discarded no other pairs), increase paired pref
        if max(rank_counts.values()) == 1 and len(suit_counts) < 3:
            self._discard_data['kept_pair_count'] += 1
            self.pref_paired = self.pref_paired * (1 - lr) + 2.0 * lr

    def reset_hand(self):
        self._hand_actions = []
        self._raised_this_hand = False
        self._hands_observed += 1

    def _total_postflop_actions(self) -> int:
        return sum(self._actions_total[1:4])

    def _total_postflop_raises(self) -> int:
        return sum(self._raises[1:4])

    def get_context(self, street: int) -> dict:
        total_actions = sum(self._actions_total)
        total_raises = sum(self._raises)
        pf_actions = self._actions_total[0]
        pf_raises = self._raises[0]

        raise_rate = total_raises / max(1, total_actions)
        pf_raise_rate = pf_raises / max(1, pf_actions)

        total_our_raises_faced = sum(self._folds_to_our_raise[s] + self._calls_to_our_raise[s] for s in range(4))
        total_folds_to_us = sum(self._folds_to_our_raise)
        fold_rate = total_folds_to_us / max(1, total_our_raises_faced)

        # Per-street fold rates: postflop (streets 1-3) separated from preflop
        postflop_raises_faced = sum(self._folds_to_our_raise[s] + self._calls_to_our_raise[s] for s in range(1, 4))
        postflop_folds_to_us = sum(self._folds_to_our_raise[s] for s in range(1, 4))
        postflop_fold_rate = postflop_folds_to_us / max(1, postflop_raises_faced)
        preflop_raises_faced = self._folds_to_our_raise[0] + self._calls_to_our_raise[0]
        preflop_fold_rate = self._folds_to_our_raise[0] / max(1, preflop_raises_faced)

        bluff_rate = self._bluff_alpha / (self._bluff_alpha + self._bluff_beta)
        avg_raise_frac = self.avg_raise_frac()

        strong_types = {'straight_flush', 'full_house', 'flush', 'straight', 'trips'}
        sd_strong_count = sum(v for k, v in self._sd_hand_types.items() if k in strong_types)
        sd_strong_rate = sd_strong_count / max(1, self._sd_count)

        raises_often = raise_rate > 0.40
        folds_often = fold_rate > 0.40

        if raises_often and not folds_often:
            opp_type = 'lag'
        elif raises_often and folds_often:
            opp_type = 'tag'
        elif not raises_often and not folds_often:
            opp_type = 'station'
        else:
            opp_type = 'nit'

        # VPIP: fraction of all hands where opponent voluntarily entered the pot
        # (took at least one non-fold preflop action). Low VPIP = very selective.
        # _actions_total[0] = preflop actions taken, _folds[0] = preflop folds
        # _hands_observed = total hands including prefolds (no opponent action)
        opp_entered = max(0, self._actions_total[0] - self._folds[0])
        vpip = opp_entered / max(1, self._hands_observed)

        return {
            'raise_rate': raise_rate,
            'pf_raise_rate': pf_raise_rate,
            'raises_often': raises_often,
            'fold_rate_to_our_raise': fold_rate,
            'postflop_fold_rate': postflop_fold_rate,
            'preflop_fold_rate': preflop_fold_rate,
            'postflop_raises_faced': postflop_raises_faced,
            'folds_often': folds_often,
            'pf_fold_rate': self._folds[0] / max(1, pf_actions),
            'vpip': vpip,
            'bluff_rate': bluff_rate,
            'avg_raise_frac': avg_raise_frac,
            'is_trap_line': self.is_trap_line(),
            'is_barrel_line': self.is_barrel_line(),
            'opponent_type': opp_type,
            'raised_this_hand': self._raised_this_hand,
            'total_actions': total_actions,
            'sd_count': self._sd_count,
            'sd_strong_rate': sd_strong_rate,
        }

    def avg_raise_frac(self) -> float:
        if len(self._raise_sizes) < 5:
            return 0.50
        return sum(sz / max(1, p) for sz, p in self._raise_sizes) / len(self._raise_sizes)

    def is_trap_line(self) -> bool:
        if len(self._hand_actions) < 2:
            return False
        return 'CHECK' in self._hand_actions[:-1] and self._hand_actions[-1] == 'RAISE'

    def is_barrel_line(self) -> bool:
        return sum(1 for a in self._hand_actions if a == 'RAISE') >= 2

    def opp_weights_fn(self, hand: tuple) -> float:
        c1, c2 = hand
        w = 1.0
        if is_suited(c1, c2):
            w *= self.pref_suited
        if is_paired(c1, c2):
            w *= self.pref_paired
        return w

    @property
    def hands_observed(self) -> int:
        """Number of hands with opponent data (used for confidence blending in v4 solver)."""
        return self._hands_observed

