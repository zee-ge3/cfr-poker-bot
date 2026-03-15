"""
Claude2Agent v37 — Adaptive Exploitative Bot

Strategy principles (no opponent-specific hardcoding):
  1. HAND TIERS: SF/FH/Flush = monster, Trips/Straight = strong,
     Two-pair/Top-pair = medium, rest = weak.
  2. VALUE BETTING: Equity-driven sizing with early-game aggression multiplier.
     Bigger bets vs calling stations; larger pots early when variance is cheap.
  3. ADAPTIVE CALL THRESHOLD: Bayesian estimate + context-aware aggr_call_floor.
     Adjusts for opponent raise frequency, bet sizing, action line (trap/barrel),
     discard pattern (flush chaser / pair keeper), and showdown hand distribution.
  4. OPPONENT MODELING: Tracks raise rate, bet sizes, per-hand action sequences,
     discard patterns across hands, and showdown hand type distribution.
  5. EARLY GAME AGGRESSION: Smooth decay curve (hands 0→300), call/bet wider
     early; protection mode (lead > 10% of remaining bleed) tightens to lock win.
  6. PROBABILISTIC THRESHOLDS: Sigmoid gray zones on all key decisions to prevent
     exploitation of hard cutoffs.
  7. PROBE SIZING: Above Bayesian-estimated call threshold; suppressed vs
     hyper-aggressive opponents (raises_often) and in protection mode.
"""

import math
import random
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv

AT          = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
DECK_SIZE   = 27
NUM_RANKS   = len(PokerEnv.RANKS)

# Tiny residual discount after MC already models opp's best-2-from-5 selection.
EQ_DISCOUNT_BASE = 0.02

# Reverse lookup: card string → our int encoding (for showdown parsing)
_CARD_STR_TO_INT = {PokerEnv.int_card_to_str(i): i for i in range(DECK_SIZE)}


class Claude2Agent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.evaluator = PokerEnv().evaluator

        # Opponent action counts
        self._opp_raise_count  = 0
        self._opp_call_count   = 0
        self._opp_check_count  = 0
        self._opp_fold_count   = 0
        self._hand_count       = 0

        # Bayesian call-threshold model
        # Each entry: (pot_odds_we_bet, True=called / False=folded)
        # Neutral prior centred at 0.50
        self._bet_outcomes: list[tuple[float, bool]] = [
            (0.42, True), (0.58, False)
        ]
        self._pending_bet_pot_odds: float | None = None

        # Per-hand / per-street state
        self._checked_this_street = False
        self._bet_this_street     = False   # True if we raised this street
        self._prev_street         = -1
        self._time_left           = 500.0
        self._phase_mult          = 1       # set once on first observation (1/2/3x)

        # Guaranteed-win lock-in (CoA strategy)
        # act(terminated=True) is never called in tournament (observe() handles it),
        # so we track both here.
        self._cumulative_reward   = 0.0
        self._hands_played        = 0
        self._total_hands         = 1000

        # Showdown & discard learning
        # _opp_showdown_scores: treys hand scores seen at showdown (lower = stronger)
        # _opp_subopt_bonus_sum: accumulated equity bonus from opp's suboptimal discards
        # _opp_subopt_count: showdowns where we had full 5-card opp data
        self._opp_showdown_scores: list = []
        self._opp_passive_sd_scores: list = []  # showdowns where opp didn't raise (unbiased tightness)
        self._opp_raised_this_hand: bool = False
        self._opp_subopt_bonus_sum: float = 0.0
        self._opp_subopt_count: int = 0

        # Bet size tracking: list of (raise_amount, pot_size) tuples
        self._opp_raise_sizes: list = []

        # Per-hand action line (e.g. ['CHECK','RAISE'] = check-raise trap)
        self._opp_hand_actions: list = []

        # Discard pattern tracking across hands
        self._opp_discard_data: dict = {
            'kept_suited_count': 0,
            'kept_pair_count': 0,
            'discarded_pair_count': 0,
            'total_observed': 0,
        }
        self._opp_discards_analyzed: bool = False  # per-hand flag to analyze once

        # Showdown hand type distribution
        self._opp_sd_hand_types: dict = {}

    def __name__(self):
        return "Claude2Agent"

    # ------------------------------------------------------------------ #
    #  Opponent model                                                      #
    # ------------------------------------------------------------------ #

    def _match_pressure(self) -> float:
        """Pressure scalar in [-1, +1].

        +1 = desperate (far behind, need variance)
         0 = neutral
        -1 = comfortable lead (protect it)

        Influences call/probe thresholds:
          +0.05 max adjustment for desperation (call wider, probe more)
          -0.05 max adjustment for comfortable lead (tighten up)
        """
        remaining = self._total_hands - self._hands_played
        if remaining <= 0:
            return 0.0
        cum = self._cumulative_reward
        # Rough bleed rate: 1.5 chips/hand. Max recovery: ~3 chips/hand aggressive.
        if cum >= 0:
            lead_ratio = cum / max(1, remaining * 1.5)
            return max(-1.0, -lead_ratio)   # comfortable lead → negative pressure
        else:
            deficit = -cum
            max_recovery = remaining * 3.0
            return min(1.0, deficit / max(1, max_recovery))  # behind → positive pressure

    def _opp_action_total(self) -> int:
        return (self._opp_raise_count + self._opp_call_count
                + self._opp_check_count + self._opp_fold_count)

    def _opp_aggression(self) -> float:
        """Raise rate among all opponent actions (proxy for value-bet frequency).

        Uses a Bayesian blend of a neutral prior (0.30) and observed raise rate,
        weighted by number of observations. This allows faster adaptation to
        aggressive opponents without using unreliable small-sample estimates.
        """
        total = self._opp_action_total()
        if total == 0:
            return 0.30
        observed_rr = self._opp_raise_count / total
        # Blend: weight=0 at total=0, weight=1 at total>=8
        weight = min(1.0, total / 8.0)
        return 0.30 + weight * (observed_rr - 0.30)

    def _opp_bluff_rate(self) -> float:
        """
        Rough bluff-rate estimate: when opp raises, fraction that are likely bluffs.
        High overall raise-rate → some bluffs mixed in.
        Low raise-rate → mostly value.
        """
        rr = self._opp_aggression()
        # If they raise >40% of actions, some must be bluffs (very wide range)
        if rr > 0.45:
            return 0.15
        if rr > 0.35:
            return 0.10
        return 0.05  # conservative: assume mostly value

    def _opp_raises_often(self) -> bool:
        """True when opponent raises a very high fraction of their actions.

        Indicates hyper-aggressive 'raise-everything' style. When True:
          - Widen preflop call range (their range is too wide to justify folding)
          - Suppress thin probes (their reraise inflates the pot vs our medium hands)
        """
        total = self._opp_action_total()
        if total < 10:
            return False
        return (self._opp_raise_count / total) > 0.48

    def _opp_folds_often(self) -> bool:
        """True when opponent folds a high fraction of the time when facing a bet.

        Primary source: _bet_outcomes, which reliably records (pot_odds, called)
        for every bet we made. called=False means opp did NOT call (folded/no-call).
        This works regardless of whether terminal observations send opp_last_action.

        Fallback: opp_fold_count / (fold+call) — only used if _bet_outcomes is thin.
        """
        # Skip the 2 neutral priors; use real observed bet responses
        real = self._bet_outcomes[2:]
        if len(real) >= 6:
            fold_responses = sum(1 for _, called in real if not called)
            return fold_responses / len(real) > 0.45

        # Fallback to action counts (may be unreliable if terminal folds not reported)
        facing = self._opp_fold_count + self._opp_call_count
        if facing < 6:
            return False  # unknown → assume calling station (bigger value bets early)
        return self._opp_fold_count / facing > 0.45

    # ------------------------------------------------------------------ #
    #  Showdown range learning                                             #
    # ------------------------------------------------------------------ #

    def _soft_thr(self, equity: float, thr: float, temp: float = 0.04) -> bool:
        """Probabilistic threshold using sigmoid curve.

        P(action) = sigmoid((equity - thr) / temp)
        At equity=thr: 50%. At thr+2*temp: ~88%. At thr-2*temp: ~12%.
        Replaces hard cutoffs with smooth transitions to prevent exploitation.
        """
        x = (equity - thr) / max(0.001, temp)
        try:
            p = 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            p = 1.0 if x > 0 else 0.0
        return random.random() < p

    def _opp_avg_showdown_score(self) -> float:
        """Average treys score at opponent showdowns. Lower = stronger hand.
        Neutral prior 3500 (mid-pair range) until we have enough data."""
        scores = self._opp_passive_sd_scores  # unbiased: opp passive hands only
        if len(scores) < 5:
            return 3500.0
        return sum(scores) / len(scores)

    def _opp_showdown_tight(self) -> bool:
        """True if opp only shows down with strong hands when passive (avg score < 2800).
        Uses only non-raising showdowns to avoid selection bias: if opp raises 70% of hands,
        showdowns skew toward their value range, not their overall calling range.
        In 27-card deck: trips=1610-2467, flush=323-1599, pair=3326-6185."""
        return self._opp_avg_showdown_score() < 2800

    def _opp_suboptimality_bonus(self) -> float:
        """Equity bonus from opp keeping suboptimal cards at discard.
        MC models opp as always keeping their best 2-card hand. When opp keeps
        a weaker hand than optimal, our real equity exceeds the MC estimate.
        Returns average bonus per hand, capped at 0.04."""
        if self._opp_subopt_count < 5:
            return 0.0
        return min(0.04, self._opp_subopt_bonus_sum / self._opp_subopt_count)

    # ------------------------------------------------------------------ #
    #  Bet size, action line, discard pattern, showdown hand type          #
    # ------------------------------------------------------------------ #

    def _opp_avg_raise_frac(self) -> float:
        """Average opponent raise as fraction of pot. <0.35 = min-raiser, >0.70 = value-heavy."""
        if len(self._opp_raise_sizes) < 5:
            return 0.50
        return sum(sz / max(1, p) for sz, p in self._opp_raise_sizes) / len(self._opp_raise_sizes)

    def _opp_line_is_trap(self) -> bool:
        """Opp checked an earlier street then raised — classic slowplay/check-raise."""
        acts = self._opp_hand_actions
        return len(acts) >= 2 and 'CHECK' in acts[:-1] and acts[-1] == 'RAISE'

    def _opp_line_is_barrel(self) -> bool:
        """Opp raised on 2+ streets this hand — persistent aggression."""
        return sum(1 for a in self._opp_hand_actions if a == 'RAISE') >= 2

    def _analyze_opp_discards(self, opp_discards: list) -> None:
        """Deduce what opponent likely kept from their 3 discarded cards."""
        from collections import Counter
        cards = [c for c in opp_discards if c != -1]
        if len(cards) != 3:
            return
        self._opp_discard_data['total_observed'] += 1
        ranks = [c % NUM_RANKS for c in cards]
        suits = [c // NUM_RANKS for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        if max(rank_counts.values()) >= 2:
            self._opp_discard_data['discarded_pair_count'] += 1
        # All 3 different suits → kept cards likely share a suit (flush draw)
        if len(suit_counts) == 3:
            self._opp_discard_data['kept_suited_count'] += 1
        # No pair in discards and not all different suits → likely kept a pair
        elif max(rank_counts.values()) == 1:
            self._opp_discard_data['kept_pair_count'] += 1

    def _opp_is_flush_chaser(self) -> bool:
        total = self._opp_discard_data['total_observed']
        return total >= 15 and self._opp_discard_data['kept_suited_count'] / total > 0.45

    def _opp_is_pair_keeper(self) -> bool:
        total = self._opp_discard_data['total_observed']
        return total >= 15 and self._opp_discard_data['kept_pair_count'] / total > 0.40

    def _opp_sd_flush_rate(self) -> float:
        """Fraction of showdowns where opp had flush or better."""
        total = sum(self._opp_sd_hand_types.values())
        if total < 8:
            return 0.20
        flush_plus = (self._opp_sd_hand_types.get('flush', 0) +
                      self._opp_sd_hand_types.get('full_house', 0) +
                      self._opp_sd_hand_types.get('straight_flush', 0))
        return flush_plus / total

    # ------------------------------------------------------------------ #
    #  Bayesian call-threshold estimation                                  #
    # ------------------------------------------------------------------ #

    def _est_call_threshold(self) -> float:
        """
        Estimate opponent's call threshold from observed bet/response history.
        Uses sorted boundary: threshold lies between max(folds) and min(calls).
        """
        calls = [po for po, c in self._bet_outcomes if c]
        folds = [po for po, c in self._bet_outcomes if not c]

        if len(calls) < 2 or len(folds) < 2:
            # Fall back to fold-rate heuristic
            total = self._opp_action_total()
            if total < 8:
                return 0.50
            fold_rate = self._opp_fold_count / total
            # Higher fold rate → higher call threshold (they're selective callers)
            return min(0.60, max(0.40, 0.42 + fold_rate * 0.60))

        upper = max(folds)   # largest pot_odds they folded to
        lower = min(calls)   # smallest pot_odds they called
        if lower >= upper:
            return max(0.38, lower - 0.03)
        est = (upper + lower) / 2.0
        self.logger.info(
            f"Bayesian call_thresh={est:.3f} "
            f"(calls={len(calls)} folds={len(folds)})"
        )
        return est

    def _record_bet(self, pot_odds: float):
        self._pending_bet_pot_odds = pot_odds

    def _resolve_bet(self, opp_action: str):
        if self._pending_bet_pot_odds is None:
            return
        po = self._pending_bet_pot_odds
        self._pending_bet_pot_odds = None
        if "RAISE" in opp_action or opp_action == "CALL":
            self._bet_outcomes.append((po, True))
        elif opp_action == "FOLD":
            self._bet_outcomes.append((po, False))
        if len(self._bet_outcomes) > 80:
            self._bet_outcomes = self._bet_outcomes[-80:]

    def _n_samples(self, base: int) -> int:
        """Scale MC samples to available time budget.

        Dynamic scaling: computes per-hand time budget from time_left / remaining_hands
        and targets 40% of that budget for this MC call. Caps scale with phase.

        phase_mult=9 (simulation): always base counts for reproducibility/speed.
        Emergency taper kicks in when time critically low (<250s remaining).
        """
        t = self._time_left
        remaining = max(1, self._total_hands - self._hands_played)
        per_hand_sec = t / remaining

        # Emergency taper: rate-based — kick in when per-hand budget is tiny.
        # Absolute t < 10s is a hard floor regardless of remaining hands.
        if t < 10 or per_hand_sec < 0.02: return max(5,  base // 8)
        if per_hand_sec < 0.05:           return max(10, base // 4)
        if per_hand_sec < 0.10:           return max(20, base // 2)

        # Simulation mode: base counts for speed
        if self._phase_mult == 9:
            return base

        # Per-hand time budget → sample count budget
        # Empirical: ~16000 samples/sec on match server → target 40% of per-hand time
        budget = int(per_hand_sec * 16000 * 0.40)

        # Phase caps: higher phase = more time/vCPU → allow more samples
        if self._phase_mult == 3:  # Phase 3 (1500s, 4 vCPU)
            return max(base, min(budget, base * 25))
        if self._phase_mult == 2:  # Phase 2 (1000s, 2 vCPU)
            return max(base, min(budget, base * 20))
        # Phase 1 (500s): cap at 15x base; budget formula handles graceful taper
        return max(base, min(budget, base * 15))

    # ------------------------------------------------------------------ #
    #  Monte Carlo equity                                                  #
    # ------------------------------------------------------------------ #

    def _board_discount(self, community_cards) -> float:
        """Tiny residual discount — MC now models opp's best-2-from-5 selection
        directly, so board-texture adjustments are no longer needed here."""
        return EQ_DISCOUNT_BASE

    def _equity(self, my_cards, community_cards, excluded, n: int = 60) -> float:
        """Monte Carlo equity.

        Opponent model: they were dealt 5 cards and KEPT the best 2 (discard
        mechanic). We simulate this by sampling 5 cards for opp and evaluating
        all C(5,2)=10 pairs, keeping the strongest. This directly models the
        selection bias — opp's range is concentrated at pairs, flushes, and
        straights rather than random 2-card hands.
        """
        shown = set(my_cards)
        for c in community_cards:
            if c != -1: shown.add(c)
        for c in excluded:
            if c != -1: shown.add(c)

        pool        = [i for i in range(DECK_SIZE) if i not in shown]
        board_known = [c for c in community_cards if c != -1]
        board_need  = 5 - len(board_known)
        need        = 5 + board_need   # 5 for opp's dealt hand + remaining board

        if need > len(pool):
            return 0.5

        my_t    = [int_to_card(c) for c in my_cards]
        board_t = [int_to_card(c) for c in board_known]
        pool_t  = [int_to_card(c) for c in pool]

        # All C(5,2) index pairs for opp's best-2-from-5 selection
        opp_pairs = list(combinations(range(5), 2))

        wins = ties = total = 0
        evaluate = self.evaluator.evaluate
        for _ in range(n):
            idx        = random.sample(range(len(pool_t)), need)
            opp_five   = [pool_t[i] for i in idx[:5]]
            full_board = board_t + [pool_t[i] for i in idx[5:]]

            # Opp selects their best 2-card hand from the 5 dealt
            best_opp = 99999
            for pi, pj in opp_pairs:
                s = evaluate([opp_five[pi], opp_five[pj]], full_board)
                if s < best_opp:
                    best_opp = s

            mr = evaluate(my_t, full_board)
            if mr < best_opp:    wins += 1
            elif mr == best_opp: ties += 1
            total += 1

        return (wins + 0.5 * ties) / total if total else 0.5

    def _preflop_best(self, my_cards, n_per: int = 20) -> float:
        best = 0.0
        for i, j in combinations(range(5), 2):
            eq = self._equity([my_cards[i], my_cards[j]], [], [], n=n_per)
            if eq > best:
                best = eq
        return best

    def _preflop_heuristic(self, my_cards) -> float:
        """O(1) preflop strength estimate from heuristic scores.

        Used when we have all 5 cards pre-discard to avoid 10x MC equity calls.
        Maps heuristic best-pair score to approximate equity (0.45–0.75).
        Pocket pairs dominate in the 27-card game.
        """
        best = max(
            self._pair_heuristic(my_cards[i], my_cards[j], [])
            for i, j in combinations(range(5), 2)
        )
        # Heuristic score ranges (from _pair_heuristic):
        #  pair of A (rank 8): 200 + 8*6 = 248    suited extras omitted for pairs
        #  pair of 2 (rank 0): 200 + 0   = 200
        #  suited A-K (rank 8+7): 8*5+7*2+40 = 94
        #  offsuit J-T (rank 5+4): 5*5+4*2+0 = 33
        if best >= 248:  return 0.74   # pair of Aces
        if best >= 242:  return 0.70   # pair of 8s/9s
        if best >= 230:  return 0.66   # pair of 5s-7s
        if best >= 200:  return 0.62   # any pair
        if best >= 85:   return 0.54   # high suited connected
        if best >= 60:   return 0.51   # high suited
        if best >= 40:   return 0.49   # suited mid
        return 0.46                    # weak offsuit

    # ------------------------------------------------------------------ #
    #  Discard                                                             #
    # ------------------------------------------------------------------ #

    def _pair_heuristic(self, c1: int, c2: int, board: list) -> float:
        """Fast hand strength estimate for pre-filtering keep pairs.

        Returns very high scores for made hands (straight/flush) so they
        are always included in the MC candidates, preventing the bug where
        pair-of-2s (score 200) beats a made straight (score ~61) in ranking.
        """
        r1, r2 = c1 % NUM_RANKS, c2 % NUM_RANKS
        s1, s2 = c1 // NUM_RANKS, c2 // NUM_RANKS
        hi, lo = max(r1, r2), min(r1, r2)

        if board:
            board_ranks = set(c % NUM_RANKS for c in board)
            all_ranks = board_ranks | {r1, r2}

            # Made flush (checked first so SF scores as flush, not plain straight)
            suit_count = {}
            for c in board:
                suit = c // NUM_RANKS
                suit_count[suit] = suit_count.get(suit, 0) + 1
            suit_count[s1] = suit_count.get(s1, 0) + 1
            suit_count[s2] = suit_count.get(s2, 0) + 1
            if max(suit_count.values()) >= 5:
                return 750  # flush (including SF) beats straight

            # Made straight: any 5 consecutive ranks in {hole cards + board}.
            # Loop all 5 possible starts (ranks 0-8 → 5 windows of width 5).
            for start in range(5):
                if set(range(start, start + 5)).issubset(all_ranks):
                    return 600  # beats any pair (max=248)
            # Wheel straight A-2-3-4-5 (ranks {0,1,2,3,8})
            if {0, 1, 2, 3, 8}.issubset(all_ranks):
                return 600

            # Made trips: one of our cards + 2 board cards of same rank,
            # OR pair in hole + 1 board card of same rank
            board_rank_count = {}
            for c in board:
                r = c % NUM_RANKS
                board_rank_count[r] = board_rank_count.get(r, 0) + 1
            if r1 == r2 and board_rank_count.get(r1, 0) >= 1:
                return 400  # pair in hole + board = trips
            if board_rank_count.get(r1, 0) >= 2 or board_rank_count.get(r2, 0) >= 2:
                return 400  # one hole card + 2 board cards of same rank = trips

            # Two-pair: each hole card pairs with a different board card
            if r1 != r2 and board_rank_count.get(r1, 0) >= 1 and board_rank_count.get(r2, 0) >= 1:
                return 350  # two-pair: beats pairs (200-248), below trips/straights/flushes

        # Pair bonus (strongest type in 27-card game without board)
        if r1 == r2: return 200 + hi * 6
        # Suited bonus (flush potential)
        suited = 40 if s1 == s2 else 0
        # Flush draw: 2 suited + 1+ matching suit on board
        if s1 == s2 and board:
            board_suit_count = sum(1 for c in board if c // NUM_RANKS == s1)
            suited += board_suit_count * 20
        # Connectivity
        gap = hi - lo
        conn = max(0, 8 - gap) * 2
        return hi * 5 + lo * 2 + suited + conn

    def _best_discard(self, my_cards, community_cards, opp_discards):
        excl = [c for c in opp_discards if c != -1]
        comm = [c for c in community_cards if c != -1]
        all_pairs = list(combinations(range(5), 2))
        # Rank by heuristic, run MC only on top-5 candidates
        scored = sorted(
            all_pairs,
            key=lambda ij: self._pair_heuristic(my_cards[ij[0]], my_cards[ij[1]], comm),
            reverse=True
        )
        candidates = scored[:3]   # Top-3 is sufficient; 4th/5th rarely win MC
        n_mc = self._n_samples(50)  # 50 samples: SE≈±0.07, reduces bad picks vs n=30 (SE≈±0.09)
        best_idx, best_eq = candidates[0], -1.0
        for i, j in candidates:
            eq = self._equity([my_cards[i], my_cards[j]], comm, excl, n=n_mc)
            if eq > best_eq:
                best_eq, best_idx = eq, (i, j)
        return best_idx, best_eq

    # ------------------------------------------------------------------ #
    #  Hand classification                                                 #
    # ------------------------------------------------------------------ #

    def _classify(self, my_cards, community_cards):
        if len(my_cards) != 2 or len(community_cards) < 3:
            return None, 'unknown'
        hand  = [int_to_card(c) for c in my_cards]
        board = [int_to_card(c) for c in community_cards]
        try:
            score = self.evaluator.evaluate(hand, board)
        except Exception:
            return None, 'unknown'
        # In 27-card deck: no 4-of-a-kind possible (only 3 suits = max 3-of-a-kind)
        # Treys scores: SF≤10, FH 167-322, Flush 323-1599, Straight 1600-1609,
        #               Trips 1610-2467, Two-pair 2468-3325, Pair 3326-6185, HC 6186+
        if score <= 1599:   tier = 'monster'    # SF / FH / Flush
        elif score <= 2467: tier = 'strong'     # Straight / Trips
        elif score <= 4000: tier = 'medium'     # Two pair / strong pair
        else:               tier = 'weak'
        return score, tier

    # ------------------------------------------------------------------ #
    #  Bet helper                                                          #
    # ------------------------------------------------------------------ #

    def _bet(self, obs, frac: float, tag: str = "BET"):
        pot      = obs["my_bet"] + obs["opp_bet"]
        min_r, max_r = obs["min_raise"], obs["max_raise"]
        # Add ±15% variation to sizing to make bet patterns harder to read
        variation = 1.0 + random.uniform(-0.15, 0.15)
        size     = max(min_r, min(max_r, int(pot * frac * variation)))
        po       = size / (size + pot) if (size + pot) > 0 else 0.0
        self._record_bet(po)
        self._bet_this_street = True
        self.logger.info(f"{tag} sz={size} pot_odds={po:.2f} s={obs['street']}")
        return (AT.RAISE.value, size, 0, 0)

    # ------------------------------------------------------------------ #
    #  Core decision                                                       #
    # ------------------------------------------------------------------ #

    def _decide(self, obs, equity: float, tier: str) -> tuple:
        valid   = obs["valid_actions"]
        street  = obs["street"]
        my_bet  = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        pot     = my_bet + opp_bet
        cost    = opp_bet - my_bet

        can_raise = bool(valid[AT.RAISE.value])
        can_call  = bool(valid[AT.CALL.value])
        can_check = bool(valid[AT.CHECK.value])
        can_fold  = bool(valid[AT.FOLD.value])

        facing   = cost > 0
        pot_odds = cost / (cost + pot) if cost > 0 else 0.0

        folds_often  = self._opp_folds_often()
        raises_often = self._opp_raises_often()
        rr           = self._opp_aggression()

        # Protection mode: when comfortably ahead, minimise variance over EV.
        # Goal is to WIN the match (binary), not maximise chips.
        # Accept ~1.5 chip/hand blind bleed; refuse marginal pots that could swing.
        # Threshold: lead > 15% of expected remaining bleed → protect the win.
        pressure    = self._match_pressure()   # +1=desperate, -1=comfortable lead
        in_protection = (pressure < -0.10)

        # ============================================================
        # EARLY GAME AGGRESSION + SMOOTH AGGRESSION CURVE
        # Early hands: variance is cheap (1000 hands remaining), early leads create
        # chokehold once protection mode activates. Play wider/bigger early.
        # Smooth ±20% jitter prevents opponents detecting phase transitions.
        # Suppressed in protection mode (already winning — minimize variance).
        # ============================================================
        hands_played = self._hands_played
        early_factor = max(0.0, 1.0 - hands_played / 300.0)
        early_factor = max(0.0, early_factor * (1.0 + random.uniform(-0.20, 0.20)))
        aggression = 0.0 if in_protection else max(early_factor, max(0.0, pressure) * 0.8)

        # Continuous fold rate for probe sizing (0.0-1.0)
        real = self._bet_outcomes[2:]
        fold_rate = (sum(1 for _, c in real if not c) / len(real)) if len(real) >= 4 else 0.35

        # Position: blind_position=1 → we are BB → out of position postflop
        oop = (obs.get("blind_position", 0) == 1)

        # Aggression-aware facing-bet call floor.
        # CRITICAL FIX: previous floor 0.40 too high — opponents exploited 75-83% fold-to-raise
        # by min-raising every flop. Floor now 0.36 (continuous) with raises_often direct drop.
        # aggression (early game / desperation): call wider when variance is acceptable.
        if rr > 0.45:
            aggr_call_floor = max(0.36, 0.50 - (rr - 0.45) * 0.56 - pressure * 0.04)
        else:
            aggr_call_floor = 0.50 + max(0.0, (0.35 - rr) * 0.22) - pressure * 0.04
        # raises_often: their range is too wide → drop threshold directly
        if raises_often:
            aggr_call_floor -= 0.06
        # Early game only: call wider before hand 300. Desperation does NOT lower this —
        # calling lighter when losing vs value-betting opponents builds bigger pots we lose.
        aggr_call_floor -= early_factor * 0.05
        aggr_call_floor = max(0.28, aggr_call_floor)  # hard floor

        # Context-aware aggr_call_floor adjustments (only when facing a real bet)
        if facing and cost > 0:
            # Bet size read: current raise vs opponent average sizing
            current_raise_frac = cost / max(1, pot - cost)
            avg_frac = self._opp_avg_raise_frac()
            if current_raise_frac < 0.30 and avg_frac < 0.40:
                # Habitual min-raiser making another min-raise = probe/bluff → call wider
                aggr_call_floor -= 0.04
            elif current_raise_frac > 0.80:
                # Big raise = likely value → fold tighter
                aggr_call_floor += 0.03

            # Action line read: trap vs multi-barrel
            if self._opp_line_is_trap():
                aggr_call_floor += 0.04  # check-raise = slowplay → respect it
            elif self._opp_line_is_barrel() and raises_often:
                aggr_call_floor -= 0.03  # hyper-raiser multi-barrel = bluff continuation

            # Board texture + discard pattern read
            comm = [c for c in obs.get("community_cards", []) if c != -1]
            if comm:
                suit_cnt: dict = {}
                rank_cnt: dict = {}
                for c in comm:
                    suit_cnt[c // NUM_RANKS] = suit_cnt.get(c // NUM_RANKS, 0) + 1
                    rank_cnt[c % NUM_RANKS]  = rank_cnt.get(c % NUM_RANKS, 0) + 1
                board_flush_heavy = max(suit_cnt.values()) >= 3
                board_paired      = max(rank_cnt.values()) >= 2

                if self._opp_is_flush_chaser():
                    if board_flush_heavy:
                        aggr_call_floor += 0.03  # raise more credible on flush board
                    else:
                        aggr_call_floor -= 0.02  # no flush potential → likely bluff
                if self._opp_is_pair_keeper() and board_paired:
                    aggr_call_floor += 0.03  # likely trips/full house on paired board

                # Showdown flush rate: historical flush frequency vs current board texture
                flush_rate = self._opp_sd_flush_rate()
                if board_flush_heavy and flush_rate > 0.35:
                    aggr_call_floor += 0.02  # frequently shows flushes; credible here
                elif board_flush_heavy and flush_rate < 0.10:
                    aggr_call_floor -= 0.02  # rarely shows flushes; likely bluffing

            aggr_call_floor = max(0.28, aggr_call_floor)  # re-apply floor after adjustments

        # Against aggressive opponents: widen the pot-odds window for calls
        # (they're bluffing more, so call down lighter at better prices)
        max_call_pot_odds = 0.28 + max(0.0, (rr - 0.40) * 0.25)  # up to 0.35 vs very aggressive

        # Dynamic re-raise thresholds — calibrated for best-2-from-5 MC scale.
        # vs passive (rr ≤ 0.40): need strong hand to re-raise (they have it when they bet)
        # vs aggressive (rr > 0.40): re-raise lighter to deny equity and punish bluffs
        # pressure: desperate (+) → lower thresholds; comfortable lead (-) → higher
        reraise_thr      = max(0.58, 0.72 - max(0.0, (rr - 0.40) * 0.30) - pressure * 0.04)
        # Tight showdown range: opp only shows up with strong hands → need stronger
        # hand to call/reraise their bets. Raise threshold by 0.04.
        if self._opp_showdown_tight():
            reraise_thr = min(reraise_thr + 0.04, 0.85)
        # Lower reraise_fold_thr: at fold_rate=0.55, reraise 0.75P is profitable at any eq>0.30.
        # EV(reraise,eq=0.50) = 0.55*(P+C) + 0.45*0.55*(P+C) = 0.80*(P+C) >> EV(call)=0.38P.
        reraise_fold_thr = max(0.40, 0.50 - max(0.0, (rr - 0.40) * 0.40) - pressure * 0.04)

        # OOP penalty: acting first is a disadvantage, probe less from BB.
        # Reduced for folders: at eq=0.51 OOP vs folder (fold_rate=0.55),
        # EV(probe 0.30P) = +0.272P — the flat 0.03 blocks these profitable probes.
        oop_extra = (0.01 if folds_often else 0.03) if oop else 0.0

        def call():  return (AT.CALL.value, 0, 0, 0)
        def check():
            self._checked_this_street = True
            return (AT.CHECK.value, 0, 0, 0)
        def fold():  return (AT.FOLD.value, 0, 0, 0)

        # ============================================================
        # PRE-FLOP
        # Pressure adjustments: desperate → raise/call wider; comfortable → tighter.
        # ============================================================
        if street == 0:
            pf_raise_thr = 0.68 - pressure * 0.03 - aggression * 0.04
            pf_open_thr  = 0.57 - pressure * 0.02 - aggression * 0.04
            pf_call_thr  = 0.47 - pressure * 0.02 - aggression * 0.06

            # Distinguish BB's forced blind from a voluntary raise.
            # When SB acts first, cost=1 (BB's blind). When BB acts after SB limp, cost=0.
            # A real raise means someone put in extra beyond the initial blinds: cost > 1.
            pf_real_raise = (cost > 1)

            # Protection mode: fold weak hands preflop; accept 1-2 chip blind cost
            # instead of risking post-flop variance. Only play solid holdings.
            if in_protection:
                pf_call_thr = max(pf_call_thr, 0.60)

            # vs hyper-aggro: their preflop range is too wide to fold against.
            # At pot_odds=0.42 (raise to 8 chips), eq=0.47: EV(call)=+0.05P > 0.
            # vs normal: tight calling (0.28) accounts for value-heavy 3-bet range.
            def pf_should_call(equity, pot_odds):
                if raises_often:
                    return equity >= pot_odds + 0.03 and equity >= pf_call_thr
                return pot_odds <= 0.28

            if equity >= pf_raise_thr:
                # In deep preflop escalation (already large pot), don't reraise
                # unless very strong — 0.68 equity is a dog against a 3/4-bet range.
                in_escalation = pf_real_raise and pot_odds > 0.35
                if can_raise and (equity >= 0.76 or not in_escalation):
                    if pf_real_raise:
                        # 3-bet: pot is ~6-12 chips → frac=0.80 gives reasonable sizing
                        return self._bet(obs, 0.80, "PF-RAISE")
                    else:
                        # Open raise (SB first act or BB check option): 4x BB = 8 chips
                        return self._bet(obs, 2.0, "PF-RAISE-OPEN")
                return call() if can_call else check()
            elif equity >= pf_open_thr:
                if pf_real_raise:
                    if not pf_should_call(equity, pot_odds):
                        return fold() if can_fold else check()
                    return call() if can_call else fold()
                # Never open-raise with medium hands: actual MC equity is 0.18-0.25,
                # building a larger pot against a calling station is –EV.
                return check() if can_check else call()
            elif equity >= pf_call_thr:
                if pf_real_raise and not pf_should_call(equity, pot_odds):
                    return fold() if can_fold else check()
                return call() if can_call else check()
            else:
                if pf_real_raise:
                    return fold() if can_fold else check()
                return check() if can_check else fold()

        # ============================================================
        # NOT FACING A BET — equity-driven lead betting
        # In 27-card deck, flushes are common → 0.80+ is the true "strong" bar.
        # Bet selectively; don't inflate pots with marginal leads.
        # ============================================================
        # Early game only: bet bigger before hand 300 to build leads.
        # Desperation does NOT increase bet sizing — bigger bets when losing = bigger pot losses.
        early_bet_mult = 1.0 + early_factor * 0.20  # up to 1.20x early, 1.0x after hand 300

        if not facing:
            if self._soft_thr(equity, 0.83 - aggression * 0.02, 0.02):
                # Near-lock: big value bet. Bet bigger vs calling stations.
                # Slow-play 10% of the time on flop/turn (not river) to vary our range
                # and build pots via check-raise or induce bluffs.
                slow_play = (street < 3
                             and equity >= 0.90
                             and folds_often
                             and random.random() < (0.20 if oop else 0.10))
                if slow_play and can_check:
                    self.logger.info(f"SLOW-PLAY s={street} eq={equity:.2f}")
                    return check()
                if street < 3:
                    frac = (0.65 if folds_often else 0.92) * early_bet_mult
                else:
                    frac = (0.88 if folds_often else 1.05) * early_bet_mult
                if can_raise:
                    return self._bet(obs, min(frac, 1.50), "VALUE-LARGE")
                return check() if can_check else call()

            elif self._soft_thr(equity, 0.63 - aggression * 0.02, 0.03) and not in_protection:
                # Strong: standard value bet (low FH qualifies; flush/trips do not).
                if street < 3:
                    frac = (0.55 if folds_often else 0.80) * early_bet_mult
                else:
                    frac = (0.75 if folds_often else 0.90) * early_bet_mult
                if can_raise:
                    return self._bet(obs, min(frac, 1.20), "VALUE-STD")
                return check() if can_check else call()

            elif street < 3 and equity >= aggr_call_floor + (0.0 if folds_often else 0.03) + oop_extra - pressure * 0.03:
                # Probe: only when equity clears aggr_call_floor (enough to call a re-raise).
                # +0.03 extra vs stations (thin value is still +EV: ~0.04P gain per probe at 0.54 eq).
                # Size: fold-equity play vs folders (small to induce calls), value play vs stations (bigger).
                # Protection/hyper-aggro: skip probe — check and let them define the price.
                if in_protection or raises_often:
                    return check() if can_check else fold()
                if folds_often:
                    frac = max(0.24, min(0.40, 0.24 + fold_rate * 0.20))
                else:
                    # Station: they'll call → bet bigger for max value extraction.
                    # Steeper formula with higher cap (0.65) vs old (0.55) extracts
                    # ~0.025x pot extra chips per probe from calling stations.
                    frac = max(0.40, min(0.65, 0.30 + equity * 0.50))
                if can_raise:
                    return self._bet(obs, frac, "PROBE")
                return check() if can_check else fold()

            elif street in (1, 2) and can_raise and folds_often and not in_protection:
                # Semi-bluff/steal: flop or turn, confirmed folders only.
                # EV(bet B) - EV(check) = fold_rate*(1-eq)*P + (1-fold_rate)*(2*eq-1)*B
                # At fold_rate=0.55, B=0.30P: break-even eq=1.48 — ALWAYS profitable vs folder.
                # Extended from equity<0.38+turn-only to ALL hands below probe threshold.
                # Flop: 60% of turn freq (more streets ahead = more conservative commitment).
                # EV always positive vs folder: higher multiplier extracts more.
                # Self-regulating: if opp calls more, fold_rate drops → folds_often flips False.
                steal_freq = min(0.65 + aggression * 0.10,
                                fold_rate * 0.90 + max(0.0, pressure * 0.08) + aggression * 0.15)
                if not oop:  steal_freq = min(0.78 + aggression * 0.07, steal_freq * 1.15)
                if street == 1:  steal_freq *= 0.60  # flop: reduce for multi-street commitment
                if random.random() < steal_freq:
                    return self._bet(obs, 0.30, "STEAL")
                return check() if can_check else fold()

            elif street in (1, 2) and can_raise and not folds_often and equity >= 0.47:
                # Thin probe vs stations below main probe threshold (equity ~0.47-0.55).
                # EV gain at any eq > 1/3 vs station is positive:
                # At eq=0.50, fold_rate=0 (never folds): EV gain = B*(3*0.50-1) = +0.175P.
                # Build value pots against calling stations; flop less aggressive (2 streets ahead).
                # vs hyper-aggro: skip — their reraise with medium hands blows up the pot -EV.
                if not raises_often:
                    freq = 0.70 if street == 2 else 0.35
                    if random.random() < freq:
                        frac = max(0.28, min(0.45, 0.28 + equity * 0.25))
                        return self._bet(obs, frac, "PROBE-STN-THIN")
                return check() if can_check else fold()

            elif street == 3 and can_raise and not in_protection:
                # Thin river value bet: trips/flush (equity ~0.48-0.63).
                # Protection mode: skip all thin river bets; check and evaluate their bet.
                if folds_often and 0.48 <= equity < 0.63:
                    # High frequency: EV gain ≈ +0.30P per bet vs confirmed folder.
                    # Fold equity + positive showdown EV both justify near-always betting.
                    freq = 0.75 if oop else 0.90
                    if random.random() < freq:
                        frac = max(0.22, min(0.35, 0.22 + fold_rate * 0.20))
                        return self._bet(obs, frac, "THIN-RIVER")
                elif not folds_often and 0.45 <= equity < 0.63:
                    if random.random() < 0.88:  # near-always vs station; OOP allowed
                        frac = max(0.28, min(0.42, 0.28 + equity * 0.20))
                        return self._bet(obs, frac, "THIN-RIVER-STN")
                elif folds_often and equity < 0.48:
                    # River bluff: fold equity math proves profitable at ANY equity vs folders.
                    # EV(bluff 0.30P) - EV(check) = fold_rate*(1-eq)*P + (1-fold_rate)*(2*eq-1)*0.30P
                    # At fold_rate=0.55, eq=0.30: +0.35P gain; break-even fold_rate < 10%.
                    # Frequency scales with fold_rate evidence; OOP slightly discounted.
                    # Higher cap: folds_often=True is self-regulating (if they call more,
                    # fold_rate drops and folds_often flips False, stopping bluffs).
                    bluff_freq = min(0.75, max(0.0, fold_rate - 0.40) * 3.0 + max(0.0, pressure * 0.10))
                    if not oop:  bluff_freq = min(0.90, bluff_freq * 1.2)
                    if random.random() < bluff_freq:
                        return self._bet(obs, 0.30, "RIVER-BLUFF")
                return check() if can_check else fold()

            else:
                return check() if can_check else fold()

        # ============================================================
        # V2: Context-aware aggr_call_floor adjustments (bet size, action line,
        # discard patterns, showdown hand types). Applied here so they affect
        # FACING A BET logic only (NOT FACING section uses unadjusted floor).
        # ============================================================
        if facing and cost > 0:
            # Bet size read: small raise from habitual min-raiser = probe/bluff
            current_raise_frac = cost / max(1, pot - cost)
            avg_frac = self._opp_avg_raise_frac()
            if current_raise_frac < 0.30 and avg_frac < 0.40:
                aggr_call_floor -= 0.04   # min-raise = likely probe
            elif current_raise_frac > 0.80:
                aggr_call_floor += 0.03   # big raise = more likely value
        # Action line: check-raise trap deserves more respect; persistent barrel from
        # hyper-raiser is more likely bluff continuation
        if self._opp_line_is_trap():
            aggr_call_floor += 0.04
        if self._opp_line_is_barrel() and raises_often:
            aggr_call_floor -= 0.03
        # Discard pattern + board texture: flush-chaser behavior on flush/non-flush boards
        board_cc = [c for c in obs.get("community_cards", []) if c != -1]
        if board_cc:
            board_suits = [c // NUM_RANKS for c in board_cc]
            flush_board = max(board_suits.count(s) for s in set(board_suits)) >= 3
            board_ranks = [c % NUM_RANKS for c in board_cc]
            paired_board = len(set(board_ranks)) < len(board_ranks)
            if self._opp_is_flush_chaser():
                aggr_call_floor += 0.03 if flush_board else -0.02
            if self._opp_is_pair_keeper() and paired_board:
                aggr_call_floor += 0.03
            # Showdown distribution: opp often shows flush on flush board = credible
            if flush_board and self._opp_sd_flush_rate() > 0.35:
                aggr_call_floor += 0.02
        aggr_call_floor = max(0.28, aggr_call_floor)  # re-apply hard floor

        # ============================================================
        # FACING A BET — conservative calling range.
        # When they bet/raise, their range is concentrated upward.
        # Only continue with genuinely strong hands or excellent price.
        # ============================================================

        river_only_value = (street == 3)

        # Rule 1: Near-lock → reraise to build pot.
        # vs folder on river: 0.90 threshold (their range is value-concentrated).
        # vs station on river: lower to 0.80 — stations bet wide, range less concentrated.
        #   EV calc: pot=20, opp bets 10; raise to 36: EV=+37.6 vs call EV=+22. Clear winner.
        if equity >= reraise_thr:
            river_raise_thr = 0.80 if not folds_often else 0.90
            if can_raise and (street < 3 or equity >= river_raise_thr):
                # vs stations: pot grows → bet bigger; vs folders: 0.90 keeps them in.
                rr_frac = 1.20 if not folds_often else 0.90
                return self._bet(obs, rr_frac, "RERAISE")
            return call() if can_call else fold()

        # Protection mode: below near-lock, only call at a very good price.
        # Refuse marginal pots — a fold costs 0, a bad call costs 50+.
        if in_protection and equity < reraise_thr:
            if equity >= 0.50 and pot_odds <= 0.28 and can_call:
                return call()
            return fold() if can_fold else check()

        # Rule 2: Strong hand → reraise or call
        if equity >= reraise_fold_thr:
            if folds_often and can_raise and not river_only_value:
                return self._bet(obs, 0.75, "RERAISE-FOLD")
            if can_call:
                return call()
            return fold() if can_fold else check()

        # Rules 3-5 evaluate independently (no elif) so fall-through works correctly.
        # Probabilistic gray zones: soft thresholds prevent exploitation of hard cutoffs.
        # Rule 3: Decent flop/turn equity at favorable price
        if self._soft_thr(equity, aggr_call_floor, 0.04) and not river_only_value:
            if pot_odds <= max_call_pot_odds and can_call:
                return call()
            # pot too expensive here; fall through to rule 4 for wider window check

        # Rule 4: Pot-odds profitable call (covers rule 3 fall-through + river)
        # FIXED: floor lowered from 0.38 to 0.33 vs aggressive opponents.
        # raises_often direct drop: their range is too wide to call tight.
        rule4_floor = max(0.33, 0.38 - max(0.0, (rr - 0.45) * 0.25))
        if raises_often:
            rule4_floor = max(0.28, rule4_floor - 0.04)
        rule4_floor -= aggression * 0.04
        rule4_thr = max(rule4_floor, 0.47 - max(0.0, (rr - 0.35) * 0.24))
        if self._soft_thr(equity, max(pot_odds + 0.05, rule4_thr), 0.03) and pot_odds <= 0.45 and can_call:
            return call()

        # Rule 5: Marginal equity at cheap price.
        # FIXED: pot_odds limit extended from 0.25 to 0.30 base, scales with rr.
        rule5_po_limit = min(0.35, 0.25 + max(0.0, (rr - 0.40) * 0.25) + early_factor * 0.05)
        if equity >= max(pot_odds + 0.04, 0.40) and pot_odds <= rule5_po_limit and can_call:
            return call()

        return fold() if can_fold else check()

    # ------------------------------------------------------------------ #
    #  Main act                                                            #
    # ------------------------------------------------------------------ #

    def act(self, observation, reward, terminated, truncated, info):
        street = observation["street"]

        if terminated:
            self._hand_count += 1
            self._checked_this_street = False
            self._bet_this_street = False
            self._prev_street = -1
            opp_last_term = str(observation.get("opp_last_action", ""))
            # Count terminal opp action — folds always end the hand so they're
            # only visible here, not in the normal action-counting path below.
            if "RAISE" in opp_last_term:
                self._opp_raise_count += 1
            elif opp_last_term == "CALL":
                self._opp_call_count += 1
            elif opp_last_term == "CHECK":
                self._opp_check_count += 1
            elif opp_last_term == "FOLD":
                self._opp_fold_count += 1
            if self._pending_bet_pot_odds is not None:
                called = "CALL" in opp_last_term or "RAISE" in opp_last_term
                self._bet_outcomes.append((self._pending_bet_pot_odds, called))
                self._pending_bet_pot_odds = None
                if len(self._bet_outcomes) > 80:
                    self._bet_outcomes = self._bet_outcomes[-80:]
            return (AT.FOLD.value, 0, 0, 0)

        # Guaranteed-win lock-in: fold every hand once the lead is guaranteed
        # to survive the remaining bleed. Bleed rate = 1.5 chips/hand
        # (SB=1 + BB=2 per 2-hand cycle). Buffer of 2 → ~2 chip final margin.
        # Use tournament hand_number if available for accuracy.
        hand_num  = info.get("hand_number", self._hands_played)
        remaining = max(0, self._total_hands - hand_num)
        threshold = 1.5 * remaining + 2
        if remaining > 0 and self._cumulative_reward > threshold:
            self.logger.info(
                f"LOCK-IN lead={self._cumulative_reward:+.0f} "
                f"remaining={remaining} threshold={threshold:.1f}"
            )
            return (AT.FOLD.value, 0, 0, 0)

        # Reset per-street state
        if street != self._prev_street:
            self._checked_this_street = False
            self._bet_this_street = False
            self._prev_street = street

        self._time_left = float(observation.get("time_left", 500.0))

        # Detect compute phase from initial time budget (set once, never changed)
        # 9999 = simulation mode (keep base sample counts for speed)
        if self._phase_mult == 1 and self._time_left > 600:
            if self._time_left > 5000:
                self._phase_mult = 9   # simulation / dev — base counts
            elif self._time_left > 1100:
                self._phase_mult = 3   # Phase 3 (1500s) — 2.5x samples
            else:
                self._phase_mult = 2   # Phase 2 (1000s) — 1.5x samples
            self.logger.info(f"Phase detected: time_left={self._time_left:.0f} → phase_mult={self._phase_mult}")

        my_cards        = [c for c in observation["my_cards"] if c != -1]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_discards    = list(observation.get("opp_discarded_cards", [-1, -1, -1]))
        valid           = observation["valid_actions"]

        # Update opponent model + resolve pending bet
        opp_last = str(observation.get("opp_last_action", ""))
        if "RAISE" in opp_last:
            self._opp_raise_count += 1
            self._opp_raised_this_hand = True
            # Bet size tracking: record raise amount relative to pot
            my_bet  = observation.get("my_bet", 0)
            opp_bet = observation.get("opp_bet", 0)
            pot     = observation.get("pot_size", my_bet + opp_bet)
            raise_sz = opp_bet - my_bet
            if pot > 0 and raise_sz > 0:
                self._opp_raise_sizes.append((raise_sz, pot))
                if len(self._opp_raise_sizes) > 200:
                    self._opp_raise_sizes = self._opp_raise_sizes[-200:]
            self._opp_hand_actions.append('RAISE')
        elif opp_last == "CALL":
            self._opp_call_count += 1
            self._opp_hand_actions.append('CALL')
        elif opp_last == "CHECK":
            self._opp_check_count += 1
            self._opp_hand_actions.append('CHECK')
        elif opp_last == "FOLD":
            self._opp_fold_count += 1

        self._resolve_bet(opp_last)

        # Once per hand: analyze opp discards to build pattern knowledge
        if not self._opp_discards_analyzed and any(c != -1 for c in opp_discards):
            self._analyze_opp_discards(opp_discards)
            self._opp_discards_analyzed = True

        # ---- DISCARD ----
        if valid[AT.DISCARD.value]:
            if len(my_cards) != 5:
                return (AT.DISCARD.value, 0, 0, 1)
            (i, j), eq = self._best_discard(my_cards, community_cards, opp_discards)
            self.logger.info(
                f"DISCARD keep=({i},{j}) eq={eq:.2f} "
                f"cards={[PokerEnv.int_card_to_str(c) for c in my_cards]}"
            )
            return (AT.DISCARD.value, 0, i, j)

        # ---- BETTING ----
        excl = [c for c in opp_discards if c != -1]

        # Analyze opponent discards once per hand on first post-flop step
        if street > 0 and not self._opp_discards_analyzed and any(c != -1 for c in opp_discards):
            self._analyze_opp_discards(opp_discards)
            self._opp_discards_analyzed = True

        if street == 0:
            if len(my_cards) == 5:
                # Fast heuristic: avoid 10 MC calls before discard decision
                equity = self._preflop_heuristic(my_cards)
            else:
                equity = self._equity(my_cards[:2], [], excl, n=self._n_samples(20))
            tier = 'preflop'
        else:
            if len(my_cards) > 2:
                my_cards = my_cards[:2]
            # River uses more samples (n=80) for accuracy on final-street decisions.
            # Flop/turn use n=40; SE≈±0.08 is acceptable since we have another street to correct.
            n_base = 80 if street == 3 else 40
            equity = self._equity(my_cards, community_cards, excl, n=self._n_samples(n_base))
            # Small residual discount on top of MC (which already models best-2-from-5)
            equity = max(0.0, equity - self._board_discount(community_cards))
            # Suboptimality bonus: if opp historically keeps weaker cards than optimal,
            # our MC equity understates our true win rate. Apply learned correction.
            equity = min(0.99, equity + self._opp_suboptimality_bonus())
            score, tier = self._classify(my_cards, community_cards)
            if tier == 'unknown':
                tier = 'strong' if equity >= 0.70 else (
                    'medium' if equity >= 0.52 else 'weak'
                )

        real = self._bet_outcomes[2:]
        fo = (sum(1 for _, c in real if not c) / len(real)) if len(real) >= 6 else -1
        self.logger.info(
            f"H{self._hand_count} s{street} tl={self._time_left:.0f} "
            f"eq={equity:.3f} tier={tier} "
            f"call_thr={self._est_call_threshold():.2f} "
            f"fold_rate={fo:.2f} folds_often={self._opp_folds_often()} "
            f"raises_often={self._opp_raises_often()} "
            f"sd_tight={self._opp_showdown_tight()} "
            f"subopt={self._opp_suboptimality_bonus():.3f} "
            f"pressure={self._match_pressure():.2f} "
            f"hole={[PokerEnv.int_card_to_str(c) for c in my_cards]} "
            f"board={[PokerEnv.int_card_to_str(c) for c in community_cards]}"
        )

        return self._decide(observation, equity, tier)

    # ------------------------------------------------------------------ #
    #  Observe                                                             #
    # ------------------------------------------------------------------ #

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated:
            self._hands_played += 1
            self._cumulative_reward += reward
            # Reset per-hand state. Must happen in terminated block (not just in
            # showdown block) so non-showdown hands (folds) also reset correctly.
            self._opp_raised_this_hand = False
            self._opp_hand_actions = []
            self._opp_discards_analyzed = False
            if reward != 0:
                self.logger.info(
                    f"{'WON' if reward > 0 else 'LOST'} {abs(reward)} chips "
                    f"| cum={self._cumulative_reward:+.0f} hands={self._hands_played}"
                )
        if "player_0_cards" in info:
            # Identify which player we are by matching our kept cards.
            # At showdown, my_cards has exactly 2 non-(-1) ints (post-discard).
            my_strs = {PokerEnv.int_card_to_str(c)
                       for c in observation.get("my_cards", []) if c != -1}
            p0_strs = set(info["player_0_cards"])
            opp_kept_strs = (info["player_1_cards"] if my_strs == p0_strs
                             else info["player_0_cards"])
            board_strs    = info.get("community_cards", [])

            self.logger.info(
                f"SHOWDOWN us={sorted(my_strs)} opp={sorted(opp_kept_strs)} "
                f"board={board_strs}"
            )

            if len(opp_kept_strs) == 2 and len(board_strs) == 5:
                try:
                    opp_kept_t = [int_to_card(_CARD_STR_TO_INT[s]) for s in opp_kept_strs]
                    board_t    = [int_to_card(_CARD_STR_TO_INT[s]) for s in board_strs]
                    score = self.evaluator.evaluate(opp_kept_t, board_t)
                    # Classify showdown hand type for distribution tracking
                    if score <= 10:      hand_type = 'straight_flush'
                    elif score <= 322:   hand_type = 'full_house'
                    elif score <= 1599:  hand_type = 'flush'
                    elif score <= 1609:  hand_type = 'straight'
                    elif score <= 2467:  hand_type = 'trips'
                    elif score <= 3325:  hand_type = 'two_pair'
                    elif score <= 6185:  hand_type = 'pair'
                    else:                hand_type = 'high_card'
                    self._opp_sd_hand_types[hand_type] = self._opp_sd_hand_types.get(hand_type, 0) + 1
                    self._opp_showdown_scores.append(score)
                    # Unbiased tightness sample: only record passive showdowns.
                    # If opp raised this hand, their showdown cards skew toward value,
                    # not their overall range — would falsely trigger showdown_tight.
                    if not self._opp_raised_this_hand:
                        self._opp_passive_sd_scores.append(score)
                    # Suboptimality check: did opp keep their best 2-card hand?
                    # opp_discarded_cards in obs = the 3 cards they threw away.
                    opp_disc_ints = [c for c in observation.get("opp_discarded_cards", [])
                                     if c != -1]
                    if len(opp_disc_ints) == 3:
                        kept_ints  = [_CARD_STR_TO_INT[s] for s in opp_kept_strs]
                        five_ints  = kept_ints + opp_disc_ints
                        best_score = min(
                            self.evaluator.evaluate(
                                [int_to_card(five_ints[i]), int_to_card(five_ints[j])],
                                board_t
                            )
                            for i, j in combinations(range(5), 2)
                        )
                        if best_score < score:
                            # Opp kept a weaker hand than optimal. Our MC equity is
                            # understated — record bonus proportional to score gap.
                            bonus = min(0.06, (score - best_score) / 7462 * 0.8)
                            self._opp_subopt_bonus_sum += bonus
                        self._opp_subopt_count += 1

                    self.logger.info(
                        f"SD_LEARN opp_score={score} "
                        f"avg={self._opp_avg_showdown_score():.0f} "
                        f"tight={self._opp_showdown_tight()} "
                        f"subopt_bonus={self._opp_suboptimality_bonus():.3f}"
                    )
                except (KeyError, Exception) as e:
                    self.logger.warning(f"SHOWDOWN parse error: {e}")
            # Reset per-hand flag after showdown data is captured
            self._opp_raised_this_hand = False
