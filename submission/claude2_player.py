"""
Claude2Agent v6 — Adaptive Value Bot

Strategy principles (no opponent-specific hardcoding):
  1. HAND TIERS: SF/FH/Flush = monster, Trips/Straight = strong,
     Two-pair/Top-pair = medium, rest = weak.
     (Flush correctly classified as ≤1599 treys score, not ≤322)
  2. VALUE BETTING: Equity-driven sizing. Bigger bets with higher equity
     to extract maximum from opponent's calling range.
  3. ADAPTIVE CALL THRESHOLD: Bayesian estimate from observed bet/response
     history. Tracks opponent's call threshold dynamically.
  4. OPPONENT BLUFF RATE: Learned from action counts. Used to tune
     facing-bet decisions (more bluff-catching when opp is loose-aggressive).
  5. RIVER DISCIPLINE: No bluffing on river (low success rate heads-up).
     Tighter calls on river (less bluff-catching needed vs passive lines).
  6. PROBE SIZING: Always probe above the Bayesian-estimated call threshold
     so probes have positive fold equity.
"""

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

        phase_mult=1: Phase 1 (≤600s) — use base counts.
        phase_mult=2: Tournament Phase 2 (600–1100s budget) — scale 1.5x.
        phase_mult=3: Simulation (9999s) — use base counts (sim speed matters).
        Emergency taper applies when time critically low.
        """
        t = self._time_left
        if t < 60:  return max(5,  base // 8)
        if t < 150: return max(10, base // 4)
        if t < 250: return max(20, base // 2)
        if self._phase_mult == 2:  # Tournament Phase 2 (1000s budget)
            return min(int(base * 1.5), 80)
        return base

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

        folds_often = self._opp_folds_often()
        rr          = self._opp_aggression()
        pressure    = self._match_pressure()   # +1=desperate, -1=comfortable lead

        # Continuous fold rate for probe sizing (0.0-1.0)
        real = self._bet_outcomes[2:]
        fold_rate = (sum(1 for _, c in real if not c) / len(real)) if len(real) >= 4 else 0.35

        # Position: blind_position=1 → we are BB → out of position postflop
        oop = (obs.get("blind_position", 0) == 1)

        # Aggression-aware facing-bet call floor.
        # Calibrated for best-2-from-5 MC equity scale where:
        #   trips ≈ 0.45-0.52, flush ≈ 0.51, low FH ≈ 0.69, high FH ≈ 0.83-0.97
        # Base floor 0.50; rises vs passive opps (strong hand range), falls vs aggressive.
        if rr > 0.45:
            aggr_call_floor = max(0.40, 0.50 - (rr - 0.45) * 0.40 - pressure * 0.04)
        else:
            aggr_call_floor = 0.50 + max(0.0, (0.35 - rr) * 0.22) - pressure * 0.04

        # Against aggressive opponents: widen the pot-odds window for calls
        # (they're bluffing more, so call down lighter at better prices)
        max_call_pot_odds = 0.28 + max(0.0, (rr - 0.40) * 0.25)  # up to 0.35 vs very aggressive

        # Dynamic re-raise thresholds — calibrated for best-2-from-5 MC scale.
        # vs passive (rr ≤ 0.40): need strong hand to re-raise (they have it when they bet)
        # vs aggressive (rr > 0.40): re-raise lighter to deny equity and punish bluffs
        # pressure: desperate (+) → lower thresholds; comfortable lead (-) → higher
        reraise_thr      = max(0.58, 0.72 - max(0.0, (rr - 0.40) * 0.30) - pressure * 0.04)
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
            pf_raise_thr = 0.68 - pressure * 0.03   # ~0.65 when desperate, ~0.71 when safe
            pf_open_thr  = 0.57 - pressure * 0.02   # ~0.55 desperate, ~0.59 safe
            pf_call_thr  = 0.47 - pressure * 0.02

            # Distinguish BB's forced blind from a voluntary raise.
            # When SB acts first, cost=1 (BB's blind). When BB acts after SB limp, cost=0.
            # A real raise means someone put in extra beyond the initial blinds: cost > 1.
            pf_real_raise = (cost > 1)

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
                    # Conservative: MC equity vs random underestimates 3-bet range advantage.
                    # Tight calling range (0.28) implicitly accounts for range-vs-range.
                    if pot_odds > 0.28:
                        return fold() if can_fold else check()
                    return call() if can_call else fold()
                # Never open-raise with medium hands: actual MC equity is 0.18-0.25,
                # building a larger pot against a calling station is –EV.
                return check() if can_check else call()
            elif equity >= pf_call_thr:
                if pf_real_raise and pot_odds > 0.28:
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
        if not facing:
            if equity >= 0.83:
                # Near-lock: big value bet. Bet bigger vs calling stations.
                # Slow-play 10% of the time on flop/turn (not river) to vary our range
                # and build pots via check-raise or induce bluffs.
                # OOP slow-play slightly more often: we'll check, they bet, we raise.
                # Only slow-play vs folders: they might bluff into our check, building pot.
                # vs stations: never slow-play — they call but don't bluff, so checking loses value.
                slow_play = (street < 3
                             and equity >= 0.90
                             and folds_often
                             and random.random() < (0.20 if oop else 0.10))
                if slow_play and can_check:
                    self.logger.info(f"SLOW-PLAY s={street} eq={equity:.2f}")
                    return check()
                if street < 3:
                    frac = 0.65 if folds_often else 0.92  # stations call → extract max on flop/turn
                else:
                    frac = 0.88 if folds_often else 1.05  # overbet river vs station (they call)
                if can_raise:
                    return self._bet(obs, frac, "VALUE-LARGE")
                return check() if can_check else call()

            elif equity >= 0.63:
                # Strong: standard value bet (low FH qualifies; flush/trips do not).
                # Calibrated for best-2-from-5 MC: low FH ≈ 0.69, flush ≈ 0.51.
                if street < 3:
                    frac = 0.55 if folds_often else 0.80  # stations call → extract max on flop/turn
                else:
                    frac = 0.75 if folds_often else 0.90  # larger river value vs station
                if can_raise:
                    return self._bet(obs, frac, "VALUE-STD")
                return check() if can_check else call()

            elif street < 3 and equity >= aggr_call_floor + (0.0 if folds_often else 0.03) + oop_extra - pressure * 0.03:
                # Probe: only when equity clears aggr_call_floor (enough to call a re-raise).
                # +0.03 extra vs stations (thin value is still +EV: ~0.04P gain per probe at 0.54 eq).
                # Size: fold-equity play vs folders (small to induce calls), value play vs stations (bigger).
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

            elif street in (1, 2) and can_raise and folds_often:
                # Semi-bluff/steal: flop or turn, confirmed folders only.
                # EV(bet B) - EV(check) = fold_rate*(1-eq)*P + (1-fold_rate)*(2*eq-1)*B
                # At fold_rate=0.55, B=0.30P: break-even eq=1.48 — ALWAYS profitable vs folder.
                # Extended from equity<0.38+turn-only to ALL hands below probe threshold.
                # Flop: 60% of turn freq (more streets ahead = more conservative commitment).
                # EV always positive vs folder: higher multiplier extracts more.
                # Self-regulating: if opp calls more, fold_rate drops → folds_often flips False.
                steal_freq = min(0.65, fold_rate * 0.90 + max(0.0, pressure * 0.08))
                if not oop:  steal_freq = min(0.78, steal_freq * 1.15)  # IP steals more
                if street == 1:  steal_freq *= 0.60  # flop: reduce for multi-street commitment
                if random.random() < steal_freq:
                    return self._bet(obs, 0.30, "STEAL")
                return check() if can_check else fold()

            elif street in (1, 2) and can_raise and not folds_often and equity >= 0.47:
                # Thin probe vs stations below main probe threshold (equity ~0.47-0.55).
                # EV gain at any eq > 1/3 vs station is positive:
                # At eq=0.50, fold_rate=0 (never folds): EV gain = B*(3*0.50-1) = +0.175P.
                # Build value pots against calling stations; flop less aggressive (2 streets ahead).
                freq = 0.70 if street == 2 else 0.35
                if random.random() < freq:
                    frac = max(0.28, min(0.45, 0.28 + equity * 0.25))
                    return self._bet(obs, frac, "PROBE-STN-THIN")
                return check() if can_check else fold()

            elif street == 3 and can_raise:
                # Thin river value bet: trips/flush (equity ~0.48-0.63).
                # Math: vs station, EV(bet) = 0.55P + 0.10B > EV(check) = 0.55P always.
                # vs folders: EV(bet) ≈ 0.80P > EV(check) ≈ 0.57P even with 55% fold rate.
                # OOP vs folder: bet forces them to call/fold at our price; check lets them
                # bet at their price and we face higher pot_odds. Betting OOP is still +EV.
                # IP 45%, OOP 30% (slightly discounted for positional disadvantage on action).
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

        # Rule 2: Strong hand → reraise or call
        if equity >= reraise_fold_thr:
            if folds_often and can_raise and not river_only_value:
                return self._bet(obs, 0.75, "RERAISE-FOLD")
            if can_call:
                return call()
            return fold() if can_fold else check()

        # Rules 3-5 evaluate independently (no elif) so fall-through works correctly.
        # Rule 3: Decent flop/turn equity at favorable price
        if equity >= aggr_call_floor and not river_only_value:
            if pot_odds <= max_call_pot_odds and can_call:
                return call()
            # pot too expensive here; fall through to rule 4 for wider window check

        # Rule 4: Pot-odds profitable call (covers rule 3 fall-through + river)
        # Widened vs aggressive opponents: high rr → wider bet range (more bluffs) →
        # our absolute equity is closer to effective equity vs their range.
        rule4_thr = max(0.38, 0.47 - max(0.0, (rr - 0.35) * 0.24))
        # Extend pot_odds limit to 0.45: at equity=0.55/pot_odds=0.40, EV=+0.62P — currently folded.
        # Cushion pot_odds+0.05 ensures we need meaningful edge above break-even before calling.
        if equity >= max(pot_odds + 0.05, rule4_thr) and pot_odds <= 0.45 and can_call:
            return call()

        # Rule 5: Marginal equity at cheap price — call if we're ahead of pot odds.
        # Extend pot_odds to 0.25: eq=0.44 at 0.23 = +0.19P; old limit missed this.
        if equity >= max(pot_odds + 0.04, 0.40) and pot_odds <= 0.25 and can_call:
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
        if self._phase_mult == 1 and self._time_left > 600:
            self._phase_mult = 3 if self._time_left > 1100 else 2
            self.logger.info(f"Phase detected: time_left={self._time_left:.0f} → phase_mult={self._phase_mult}")

        my_cards        = [c for c in observation["my_cards"] if c != -1]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_discards    = list(observation.get("opp_discarded_cards", [-1, -1, -1]))
        valid           = observation["valid_actions"]

        # Update opponent model + resolve pending bet
        opp_last = str(observation.get("opp_last_action", ""))
        if "RAISE" in opp_last:
            self._opp_raise_count += 1
        elif opp_last == "CALL":
            self._opp_call_count += 1
        elif opp_last == "CHECK":
            self._opp_check_count += 1
        elif opp_last == "FOLD":
            self._opp_fold_count += 1

        self._resolve_bet(opp_last)

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
            if reward != 0:
                self.logger.info(
                    f"{'WON' if reward > 0 else 'LOST'} {abs(reward)} chips "
                    f"| cum={self._cumulative_reward:+.0f} hands={self._hands_played}"
                )
        if "player_0_cards" in info:
            self.logger.info(
                f"SHOWDOWN us={info.get('player_0_cards')} "
                f"opp={info.get('player_1_cards')} board={info.get('community_cards')}"
            )
