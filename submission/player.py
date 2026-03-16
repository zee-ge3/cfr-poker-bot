"""
AggressiveExploitativeAgent v18r — Claude Code (Sonnet 4.6)

v18r: Revert to v18 core approach. Tournament record: v18=172/229 (75%), v30=1/4 (25%).
  All changes from v22-v32 (flush_discount, PAIR-BOARD-RAISE-DISC, FLUSH-BOARD-DISC,
  PAIRED-FLUSH-DISC, big-pot safety floors) caused progressive over-folding in tournament.

  Kept from post-v18:
  - Raise war cap (v22): prevents pot escalation in FH/SF coolers without causing folds
  - Uniform raise frac 0.75 (v27): simple, extracts more value at later streets

  Stripped (all caused over-folding):
  - flush_discount system (v22-v32): PA folded winning flushes to weaker hands
  - PAIRED-FLUSH-DISC (v26-v30): too broad, fires on ~48% of boards
  - PAIR-BOARD-RAISE-DISC (v28): over-folded Trips/TwoPair on paired boards
  - FLUSH-BOARD-DISC (v30): over-folded non-flush hands on suited boards
  - Big-pot safety floors (pot>35→0.52, pot>60→0.62): forced high thresholds, folded winners
  - Big-pot raise floor (pot>60→0.78): blocked profitable raises
  - Big-pot value bet floor (pot>40→0.62): missed value extraction

  v18's philosophy: trust MC equity estimation. The best-2-of-5 opponent model already
  accounts for opponent hand strength. Additional discounts double-count opponent range
  strength and cause net-negative over-folding.
"""
import random
from itertools import combinations

from agents.agent import Agent
from gym_env import PokerEnv
from treys import Card

AT          = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
DECK_SIZE   = 27
NUM_RANKS   = len(PokerEnv.RANKS)
NUM_SUITS   = len(PokerEnv.SUITS)

PRIOR_ALPHA = 3.5
PRIOR_BETA  = 6.5


def rank(c): return c % NUM_RANKS
def suit(c): return c // NUM_RANKS


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.evaluator = PokerEnv().evaluator

        self._hand_count  = 0
        self._cumulative  = 0.0
        self._prev_street = -1
        self._in_position = False     # SB = IP in heads-up postflop

        # Raise war tracking: set True when PA raises on current street.
        # Used to prevent re-raising into C2's re-raise (C2 never bluffs).
        self._pa_raised_this_street = False

        # Per-street raise model (Bayesian)
        self._raise_beta  = {s: [PRIOR_ALPHA, PRIOR_BETA] for s in range(4)}

        # Global fold tracking (for bluff decisions)
        self._opp_fold_n   = 0        # opponent folds observed
        self._opp_action_n = 0        # total opponent actions observed

        # Time budget (detected from first observation)
        self._time_left   = 500.0
        self._time_budget = None      # set once on first act() call

        # ── Showdown-based opponent modeling ──
        # Track opponent hand class when they raised, by street
        # Used to calibrate call thresholds: "when opp raises on street S, how strong are they?"
        self._opp_raise_showdown = {s: [] for s in range(4)}  # street → [hand_class, ...]

        # Per-hand tracking (reset each hand in observe())
        self._hand_opp_raised_streets = set()   # streets where opp raised this hand
        self._hand_discard_tracked = False       # whether we've logged this hand's discards

        # ── Bet sizing tracking ──
        # (raise_fraction_of_pot, hand_class_at_showdown) — filled at showdown
        self._opp_sizing_data = []  # list of (frac, hand_class)
        self._hand_opp_raise_fracs = []  # per-hand raise fracs (reset each hand)

        # ── Discard tendency tracking ──
        # Track what suits/ranks opponent discards → infer what they keep
        self._opp_discard_suit_counts = [0, 0, 0]  # suits discarded
        self._opp_discard_rank_counts = [0] * NUM_RANKS  # ranks discarded
        self._opp_discard_total = 0  # hands where we observed discards

        # ── Showdown keep-model: what the opponent actually kept ──
        # Track (kept_paired, kept_suited, kept_connected, avg_kept_rank)
        self._opp_kept_paired = 0     # times opponent kept a pair
        self._opp_kept_suited = 0     # times opponent kept suited cards
        self._opp_kept_connected = 0  # times kept cards within 2 ranks
        self._opp_kept_high = 0       # times both kept cards rank >= 6 (8+)
        self._opp_showdown_count = 0  # total showdowns observed

    # ── Adaptive sample count ─────────────────────────────────────────────────

    def _n_samples(self, base: int) -> int:
        """Scale samples with detected time budget; taper only when critically low."""
        # Scale base with budget: 500s→1x, 1000s→2x, ≥5000s→4x (capped)
        if self._time_budget is not None:
            if self._time_budget >= 5000:
                base = min(base * 4, base * 4)
            elif self._time_budget >= 800:
                base = base * 2

        t = self._time_left
        if t < 30:  return max(8,  base // 8)
        if t < 60:  return max(15, base // 4)
        if t < 100: return max(25, base // 2)
        return base

    # ── Core equity with best-2-of-5 opponent model (streets 0-1) ─────────────

    def _equity(self, my_cards, board, excluded, n=200):
        """Monte Carlo equity: opponent selects best 2 from 5 dealt cards.

        Used pre-discard (streets 0-1) when opponent still has 5 cards.
        Top-3 heuristic pre-filter avoids evaluating all C(5,2)=10 pairs per sample.
        """
        n = self._n_samples(n)
        shown = (set(my_cards)
                 | {c for c in board     if c != -1}
                 | {c for c in excluded  if c != -1})
        pool       = [i for i in range(DECK_SIZE) if i not in shown]
        clean      = [c for c in board if c != -1]
        board_need = 5 - len(clean)
        need       = 5 + board_need    # 5 opp cards + remaining board

        if need > len(pool):
            return 0.5

        my_mapped  = [int_to_card(c) for c in my_cards]
        pool_sz    = len(pool)
        evaluate   = self.evaluator.evaluate
        all_pairs  = list(combinations(range(5), 2))

        wins = ties = total = 0
        for _ in range(n):
            idx    = random.sample(range(pool_sz), need)
            opp5   = [pool[k] for k in idx[:5]]       # raw ints for heuristic
            opp5_t = [int_to_card(pool[k]) for k in idx[:5]]  # treys for evaluator
            board_t = ([int_to_card(c) for c in clean]
                       + [int_to_card(pool[k]) for k in idx[5:]])

            # Pre-filter: rank all 10 pairs by heuristic, evaluate top-3 only
            top3 = sorted(all_pairs,
                          key=lambda ij: self._hand_heuristic(opp5[ij[0]], opp5[ij[1]]),
                          reverse=True)[:3]

            best_opp = min(evaluate([opp5_t[pi], opp5_t[pj]], board_t)
                           for pi, pj in top3)

            mr = evaluate(my_mapped, board_t)
            if mr < best_opp:    wins += 1
            elif mr == best_opp: ties += 1
            total += 1

        return (wins + 0.5 * ties) / total if total else 0.5

    # ── Post-discard equity: opponent's 3 discards are known (streets 2-3) ─────

    def _opp_keep_weight(self, c1, c2):
        """Weight a candidate opponent keep-pair based on showdown history.

        Uses learned opponent preferences (suited, paired, connected, high)
        to predict how likely they are to hold this specific 2-card hand.
        Returns a weight >= 0.1 (never fully exclude any hand).
        Falls back to 1.0 if insufficient data.
        """
        if self._opp_showdown_count < 15:
            return 1.0  # not enough data yet

        n = self._opp_showdown_count
        r0, r1 = rank(c1), rank(c2)
        s0, s1 = suit(c1), suit(c2)

        weight = 1.0

        # Pair preference
        pair_rate = self._opp_kept_paired / n
        if r0 == r1:
            weight *= max(0.5, pair_rate * 5)   # boost if they like pairs
        # Note: don't penalize non-pairs — most hands aren't pairs

        # Suited preference
        suited_rate = self._opp_kept_suited / n
        # Baseline expectation for suited: ~33% (3 suits, random)
        if s0 == s1:
            weight *= max(0.5, suited_rate / 0.33)  # >1 if they prefer suited
        else:
            # If they keep suited 60%, off-suit gets penalized
            weight *= max(0.3, (1.0 - suited_rate) / 0.67)

        # High card preference
        high_rate = self._opp_kept_high / n
        if r0 >= 6 and r1 >= 6:
            weight *= max(0.5, high_rate / 0.15)   # baseline ~15% both high
        elif r0 < 4 and r1 < 4:
            # Both low cards — unlikely if they prefer high
            weight *= max(0.3, (1.0 - high_rate) / 0.85)

        return max(0.1, weight)

    def _equity_post_discard(self, my_cards, board, excluded, opp_discards, n=200):
        """Monte Carlo equity after opponent has discarded.

        We know the opponent's 3 discards. Their original hand was {3 discards + 2 unknown}.
        Deal 2 random cards as the unknown keeps. Use showdown-learned opponent
        preferences to weight samples (importance sampling: suited/paired/high-card bias).
        """
        n = self._n_samples(n)
        shown = (set(my_cards)
                 | {c for c in board     if c != -1}
                 | {c for c in excluded  if c != -1})
        pool       = [i for i in range(DECK_SIZE) if i not in shown]
        clean      = [c for c in board if c != -1]
        board_need = 5 - len(clean)
        need       = 2 + board_need    # 2 opp unknowns + remaining board

        if need > len(pool):
            return 0.5

        my_mapped  = [int_to_card(c) for c in my_cards]
        pool_sz    = len(pool)
        evaluate   = self.evaluator.evaluate
        opp_disc_raw = list(opp_discards)
        all_pairs    = list(combinations(range(5), 2))
        has_keep_model = self._opp_showdown_count >= 15

        w_wins = w_ties = w_total = 0.0
        for _ in range(n):
            idx    = random.sample(range(pool_sz), need)
            kept0, kept1 = pool[idx[0]], pool[idx[1]]

            # Opponent's original 5 cards: 3 discards + 2 random keeps
            opp5   = opp_disc_raw + [kept0, kept1]
            opp5_t = [int_to_card(c) for c in opp5]
            board_t = ([int_to_card(c) for c in clean]
                       + [int_to_card(pool[k]) for k in idx[2:]])

            # Pre-filter: top-3 heuristic pairs, evaluate those
            top3 = sorted(all_pairs,
                          key=lambda ij: self._hand_heuristic(opp5[ij[0]], opp5[ij[1]]),
                          reverse=True)[:3]

            best_opp = min(evaluate([opp5_t[pi], opp5_t[pj]], board_t)
                           for pi, pj in top3)

            # Weight this sample by how likely the opponent keeps this pair
            w = self._opp_keep_weight(kept0, kept1) if has_keep_model else 1.0

            mr = evaluate(my_mapped, board_t)
            if mr < best_opp:    w_wins += w
            elif mr == best_opp: w_ties += w
            w_total += w

        return (w_wins + 0.5 * w_ties) / w_total if w_total > 0 else 0.5

    # ── Heuristic hand ranker (O(1), used for discard pre-filter) ─────────────

    def _hand_heuristic(self, c1, c2):
        r1, r2 = rank(c1), rank(c2)
        s1, s2 = suit(c1), suit(c2)
        hi, lo = max(r1, r2), min(r1, r2)
        if r1 == r2:
            return 200 + hi * 6
        suited      = 40 if s1 == s2 else 0
        connectivity = max(0, 8 - (hi - lo)) * 2
        return hi * 5 + lo * 2 + suited + connectivity

    # ── Best 2-card keep from 5-card hand ─────────────────────────────────────

    def _best_discard(self, cards, board, excluded):
        all_pairs = list(combinations(range(len(cards)), 2))
        scored = sorted(all_pairs,
                        key=lambda ij: self._hand_heuristic(cards[ij[0]], cards[ij[1]]),
                        reverse=True)
        candidates = scored[:3]    # top-3 (was 5); offset the 3x MC cost per sample

        n_mc = self._n_samples(300)  # base 300; 67s→~300s at 500s budget
        best_idx, best_eq = candidates[0], -1.0
        for i, j in candidates:
            eq = self._equity([cards[i], cards[j]], board, excluded, n=n_mc)
            if eq > best_eq:
                best_eq, best_idx = eq, (i, j)
        return best_idx, best_eq

    # ── Opponent model ─────────────────────────────────────────────────────────

    def _update_opp_model(self, street: int, opp_action: str, pot: int = 0):
        if "RAISE" in opp_action:
            self._raise_beta[street][0] += 1
            self._opp_action_n += 1
            self._hand_opp_raised_streets.add(street)
            # Parse raise amount for sizing analysis
            # opp_action format: "RAISE_<amount>" or just "RAISE"
            try:
                parts = opp_action.split("_")
                if len(parts) >= 2:
                    raise_amt = int(parts[1])
                    frac = raise_amt / max(pot, 1) if pot > 0 else 0.5
                    self._hand_opp_raise_fracs.append(frac)
            except (ValueError, IndexError):
                pass
        elif opp_action in ("CALL", "CHECK"):
            self._raise_beta[street][1] += 1
            self._opp_action_n += 1
        elif opp_action == "FOLD":
            self._raise_beta[street][1] += 1
            self._opp_fold_n   += 1
            self._opp_action_n += 1
        # DISCARD and empty strings: ignore

    def _street_raise_rate(self, street: int) -> float:
        a, b = self._raise_beta[street]
        return a / (a + b)

    def _global_raise_rate(self) -> float:
        ta = sum(self._raise_beta[s][0] for s in range(4))
        tb = sum(self._raise_beta[s][1] for s in range(4))
        return ta / (ta + tb)

    def _fold_rate(self) -> float:
        if self._opp_action_n < 8:
            return 0.20    # cold prior
        return self._opp_fold_n / self._opp_action_n

    def _opp_type(self) -> str:
        """Classify opponent as lag / nit / station / tag.

        Key insight: an opponent with very low fold rate + moderate raise rate
        is LAG (playing 95%+ of hands aggressively), not station/tag.
        Station = plays many hands but PASSIVELY (low rr). LAG = plays many + raises.
        """
        rr = self._global_raise_rate()
        fr = self._fold_rate()
        if rr > 0.40:                return "lag"
        # Low fold + moderate raise = LAG (plays everything, raises often)
        # Alan Keating pattern: fr=0.01, rr=0.35 → this is LAG not station
        if fr < 0.08 and rr > 0.28:  return "lag"
        if fr > 0.40:                return "nit"
        if fr < 0.05 and rr < 0.20: return "station"  # never-folder, passive
        if fr < 0.10 and rr < 0.20: return "station"  # classic passive station
        return "tag"

    # ── Showdown-calibrated opponent raise strength ─────────────────────────────

    def _opp_raise_strength(self, street: int) -> float | None:
        """Return average hand class when opponent raised on this street at showdown.

        Lower = stronger (1=SF, 3=FH, 7=TwoPair, 9=HighCard).
        Returns None if insufficient data (<5 samples).
        """
        data = self._opp_raise_showdown.get(street, [])
        if len(data) < 5:
            return None
        return sum(data) / len(data)

    def _opp_bet_sizing_strength(self) -> float | None:
        """Return correlation between bet size and hand strength.

        Positive = bigger bets = stronger hands (exploitable: fold to big bets).
        Negative = bigger bets = weaker hands (exploitable: call big bets).
        Returns None if insufficient data.
        """
        if len(self._opp_sizing_data) < 10:
            return None
        # Average hand class for big bets (>0.6 pot) vs small bets (<0.4 pot)
        big = [hc for frac, hc in self._opp_sizing_data if frac > 0.6]
        small = [hc for frac, hc in self._opp_sizing_data if frac < 0.4]
        if len(big) < 3 or len(small) < 3:
            return None
        avg_big = sum(big) / len(big)
        avg_small = sum(small) / len(small)
        # Negative means big bets have lower hand class = stronger hands
        return avg_big - avg_small

    # ── Discard-based opponent range inference ────────────────────────────────

    def _opp_suit_preference(self) -> list[float] | None:
        """Return opponent's suit keep-preference based on discard history.

        If opponent discards a suit frequently, they're NOT keeping it.
        Returns [pref_s0, pref_s1, pref_s2] where higher = more likely to keep.
        None if insufficient data.
        """
        if self._opp_discard_total < 10:
            return None
        # Expected discard rate per suit = 1.0 (3 discards from 5 cards, 3 suits)
        # If they discard suit X more than expected, they don't like keeping it
        prefs = []
        for s in range(NUM_SUITS):
            expected = self._opp_discard_total  # expect ~1 discard per suit per hand
            actual = self._opp_discard_suit_counts[s]
            # Higher ratio = discards this suit more = prefers NOT keeping it
            pref = expected / max(actual, 1)  # >1 means they keep this suit
            prefs.append(pref)
        return prefs

    # ── Match standing ─────────────────────────────────────────────────────────

    def _match_pressure(self, remaining_hands: int) -> float:
        """Pressure scalar: +1 = desperate (need aggression), -1 = sitting pretty.

        Based on chip deficit relative to remaining hands and max recovery rate.
        Positive pressure → loosen calls, raise more, bluff more.
        Negative pressure → tighten, protect lead, avoid variance.
        """
        if remaining_hands <= 0:
            return 0.0
        cum = self._cumulative
        # Rough max recovery: ~3 chips/hand aggressive play
        max_recovery = remaining_hands * 3.0
        if cum >= 0:
            # Ahead: how comfortable is the lead?
            lead_ratio = cum / max(1, remaining_hands * 1.5)
            return max(-1.0, -lead_ratio)   # more negative = more comfortable lead
        else:
            # Behind: how desperate?
            deficit = -cum
            desperation = deficit / max(1, max_recovery)
            return min(1.0, desperation)

    # ── Call threshold ─────────────────────────────────────────────────────────

    def _call_threshold(self, pot_odds: float, street: int,
                        pressure: float = 0.0) -> float:
        """Pot-odds-aware call threshold.

        v20 calibration: v18's random-2-card equity was inflated by ~0.25 vs true
        equity. v18 floors (0.47/0.52/0.58/0.65/0.72) shifted -0.25 → new floors
        0.22/0.27/0.33/0.40/0.47. We use slightly higher floors for safety margin.
        """
        if pot_odds < 0.15:
            base = max(pot_odds + 0.06, 0.28)
        elif pot_odds < 0.25:
            base = max(pot_odds + 0.07, 0.33)
        elif pot_odds < 0.33:
            base = max(pot_odds + 0.08, 0.38)
        elif pot_odds < 0.45:
            base = max(pot_odds + 0.10, 0.44)
        else:
            base = max(pot_odds + 0.12, 0.50)

        # Opponent-type adjustment — scaled by raise rate intensity
        opp = self._opp_type()
        rr  = self._global_raise_rate()
        sr  = self._street_raise_rate(street)
        if opp == "lag":
            # Scale with how extreme the LAG is: rr=0.45 → -0.05, rr=0.70 → -0.12
            lag_adj = min(0.15, 0.05 + max(0.0, rr - 0.45) * 0.28)
            base -= lag_adj
        elif opp == "nit":
            base += 0.05    # rare bets = real hands = fold more
        else:
            # Per-street Bayesian adjustment for tag/station
            if sr < 0.30:
                base = min(base + 0.03, 0.60)
            elif sr < 0.40:
                base = min(base + 0.01, 0.55)

        # Match-pressure adjustment: desperate → call more; leading → fold more
        base -= pressure * 0.06

        # Showdown-calibrated adjustment: if we've seen enough data about what
        # opponent raises with on this street, adjust threshold accordingly.
        # Low avg_hc = opponent raises with strong hands → tighten
        # High avg_hc = opponent raises with weak hands → loosen
        opp_str = self._opp_raise_strength(street)
        if opp_str is not None:
            # Baseline expectation: ~5.5 (Straight/TwoPair)
            # If they raise with avg FH (3.0), tighten by +0.06
            # If they raise with avg HighCard (8.0), loosen by -0.06
            strength_adj = (5.5 - opp_str) * 0.025
            base += strength_adj

        return max(pot_odds + 0.02, base)

    # ── Raise sizing ──────────────────────────────────────────────────────────

    def _raise_size(self, pot: float, frac: float, min_r: int, max_r: int) -> int:
        variation = 1.0 + random.uniform(-0.20, 0.20)
        return max(min_r, min(max_r, int(pot * frac * variation)))

    # ── Pre-flop action ───────────────────────────────────────────────────────

    def _act_preflop(self, obs, my_cards, pressure: float):
        valid        = obs["valid_actions"]
        my_bet       = obs["my_bet"]
        opp_bet      = obs["opp_bet"]
        min_r        = obs["min_raise"]
        max_r        = obs["max_raise"]
        pot          = my_bet + opp_bet
        cost         = opp_bet - my_bet
        facing_raise = cost > 0

        _, best_eq = self._best_discard(my_cards, [], [])
        pf_rr      = self._street_raise_rate(0)
        opp        = self._opp_type()
        self.logger.info(f"PRE-FLOP best_eq={best_eq:.2f} pf_rr={pf_rr:.2f} opp={opp} pressure={pressure:+.2f}")

        # Raise war: pot > 40 (was 12); premiums (eq >= 0.69) always 3-bet
        is_premium   = best_eq >= 0.69
        in_raise_war = pot > 40 and facing_raise and not is_premium

        # Raise threshold: 0.52 (calibrated for true v19 equity scale), loosen vs lag
        raise_eq = 0.52
        if opp == "lag":   raise_eq -= 0.04
        elif opp == "nit": raise_eq += 0.04
        raise_eq -= pressure * 0.04   # desperate → raise wider

        if best_eq >= raise_eq and valid[AT.RAISE.value] and not in_raise_war:
            if best_eq >= 0.69 and random.random() < 0.10:
                self.logger.info(f"PRE-FLOP SLOW eq={best_eq:.2f}")
                return (AT.CALL.value, 0, 0, 0) if valid[AT.CALL.value] else (AT.CHECK.value, 0, 0, 0)
            amt = self._raise_size(max(pot, 2), 0.80, min_r, max_r)
            self.logger.info(f"PRE-FLOP RAISE eq={best_eq:.2f}")
            return (AT.RAISE.value, amt, 0, 0)

        if facing_raise:
            # Pot-odds-aware threshold — fold only hands below pot-odds + small margin
            pot_odds = cost / (cost + pot) if (cost + pot) > 0 else 0.25
            pf_thresh = pot_odds + 0.05   # very small margin above break-even
            pf_thresh = max(pf_thresh, 0.26)  # absolute floor
            if opp == "nit":
                pf_thresh = min(pf_thresh + 0.06, 0.44)
            elif opp == "lag":
                # Scale with how extreme: pf_rr=0.45 → -0.06, pf_rr=0.90 → -0.20
                lag_pf_adj = min(0.22, 0.06 + max(0.0, pf_rr - 0.45) * 0.35)
                pf_thresh = max(pf_thresh - lag_pf_adj, 0.14)
            pf_thresh -= pressure * 0.05
            if best_eq >= pf_thresh and valid[AT.CALL.value]:
                return (AT.CALL.value, 0, 0, 0)
            elif valid[AT.FOLD.value]:
                self.logger.info(f"PRE-FLOP FOLD eq={best_eq:.2f} thresh={pf_thresh:.2f} odds={pot_odds:.2f}")
                return (AT.FOLD.value, 0, 0, 0)

        if valid[AT.CHECK.value]: return (AT.CHECK.value, 0, 0, 0)
        if valid[AT.CALL.value]:  return (AT.CALL.value,  0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    # ── Post-flop betting ─────────────────────────────────────────────────────

    def _act_betting(self, obs, equity, pressure: float):
        valid        = obs["valid_actions"]
        street       = obs["street"]
        my_bet       = obs["my_bet"]
        opp_bet      = obs["opp_bet"]
        min_r        = obs["min_raise"]
        max_r        = obs["max_raise"]
        pot          = my_bet + opp_bet
        cost         = opp_bet - my_bet
        pot_odds     = cost / (cost + pot) if cost > 0 else 0.0
        facing_raise = cost > 0

        eq = equity

        # Raise war cap (v22): when C2 re-raises PA's raise, never re-raise back.
        # C2 doesn't bluff; its re-raise signals a strong hand (FH, SF, etc.).
        # PA should call or fold based on pot odds, not escalate further.
        c2_reraising_pa = facing_raise and self._pa_raised_this_street
        if c2_reraising_pa:
            self.logger.info(f"RAISE-WAR-CAP s={street} eq={eq:.2f} pot={pot:.0f}")

        # Bet sizing tell: if we've learned that big bets = strong hands,
        # adjust equity when facing a large raise
        sizing_adj = 0.0
        if facing_raise and cost > 0:
            raise_frac = cost / max(pot - cost, 1)  # raise relative to pot before raise
            sizing_tell = self._opp_bet_sizing_strength()
            if sizing_tell is not None and abs(sizing_tell) > 1.0:
                # sizing_tell < 0 means big bets = strong (lower hand class number)
                if sizing_tell < -1.0 and raise_frac > 0.6:
                    sizing_adj = 0.03   # big bet + tells strong → tighten
                elif sizing_tell < -1.0 and raise_frac < 0.3:
                    sizing_adj = -0.02  # small bet + tells weak → loosen
                elif sizing_tell > 1.0 and raise_frac > 0.6:
                    sizing_adj = -0.02  # big bet but tells = bluffs → loosen
                elif sizing_tell > 1.0 and raise_frac < 0.3:
                    sizing_adj = 0.02   # small bet but tells = value → tighten

        ip   = self._in_position
        opp  = self._opp_type()
        fr   = self._fold_rate()

        # ── Raise threshold ──
        if facing_raise:
            raise_thresh = 0.71 if pot > 25 else 0.68
        else:
            raise_thresh = 0.57
        if opp == "lag":   raise_thresh -= 0.04
        elif opp == "nit": raise_thresh += 0.04
        raise_thresh -= pressure * 0.05    # desperate → re-raise more

        if eq >= raise_thresh and valid[AT.RAISE.value] and not c2_reraising_pa:
            if eq >= 0.75 and not facing_raise and random.random() < 0.10:
                self.logger.info(f"SLOW-PLAY s={street} eq={eq:.2f}")
                return (AT.CHECK.value, 0, 0, 0)
            frac = 0.75   # uniform across all streets (v27)
            if opp == "station": frac += 0.10   # extract max vs callers
            elif opp == "nit":   frac -= 0.10   # don't bloat vs tight
            amt = self._raise_size(pot, frac, min_r, max_r)
            self.logger.info(f"RAISE s={street} eq={eq:.2f} amt={amt} opp={opp}")
            self._pa_raised_this_street = True
            return (AT.RAISE.value, amt, 0, 0)

        # ── Call ──
        if facing_raise:
            call_thresh = self._call_threshold(pot_odds, street, pressure) + sizing_adj
            if eq >= call_thresh and valid[AT.CALL.value]:
                self.logger.info(f"CALL s={street} eq={eq:.2f} odds={pot_odds:.2f} thresh={call_thresh:.2f} sizing_adj={sizing_adj:+.2f}")
                return (AT.CALL.value, 0, 0, 0)

        # ── Fold-exploitation probe (not facing raise, opponent folds often) ──
        # When opponent folds >60% to post-flop raises, raising is hugely +EV
        # even with weak hands. This is the #1 priority fix from tournament analysis.
        if not facing_raise and valid[AT.RAISE.value] and fr > 0.40:
            # Calculate minimum equity needed for a profitable probe
            # EV = fr * pot - (1-fr) * raise_cost ≥ 0
            # With 0.50x pot sizing: raise_cost ≈ 0.50 * pot
            # Break-even eq when called: need eq > (1-fr)*0.5*pot / ((1-fr)*pot) = 0.5*(1-fr)/(1-fr)
            # Simplified: when fr > 0.60, even equity = 0.15 is profitable
            if fr > 0.70:
                probe_eq_thresh = 0.15   # nearly any hand: 70%+ fold = free money
            elif fr > 0.60:
                probe_eq_thresh = 0.25   # most hands profitable
            elif fr > 0.50:
                probe_eq_thresh = 0.35   # decent hands
            else:
                probe_eq_thresh = 0.45   # marginal but still exploitable

            if eq >= probe_eq_thresh:
                # Use smaller sizing for exploitation probes (cheaper bluffs)
                probe_frac = 0.50 if fr > 0.60 else 0.60
                amt = self._raise_size(pot, probe_frac, min_r, max_r)
                self.logger.info(f"FOLD-EXPLOIT s={street} eq={eq:.2f} fr={fr:.2f} thresh={probe_eq_thresh:.2f} amt={amt}")
                self._pa_raised_this_street = True
                return (AT.RAISE.value, amt, 0, 0)

        # ── Value bet (not facing raise) ──
        if not facing_raise and valid[AT.RAISE.value]:
            value_thresh = 0.54 if ip else 0.60
            value_thresh -= pressure * 0.04    # desperate → value bet wider
            if eq >= value_thresh:
                if opp == "station":
                    amt = self._raise_size(pot, 0.80, min_r, max_r)
                    self.logger.info(f"VALUE-BET-STATION s={street} eq={eq:.2f} amt={amt}")
                else:
                    frac = 0.60 if ip else 0.50
                    amt  = self._raise_size(pot, frac, min_r, max_r)
                    self.logger.info(f"VALUE-BET s={street} eq={eq:.2f} IP={ip} amt={amt}")
                self._pa_raised_this_street = True
                return (AT.RAISE.value, amt, 0, 0)

        # ── Bluff (not facing raise, weak equity, opponent folds) ──
        if not facing_raise and valid[AT.RAISE.value]:
            bluff_fr_thresh = max(0.25, 0.40 - pressure * 0.10)
            if (fr > bluff_fr_thresh and ip and eq < 0.38 and pot < 35
                    and opp != "station"):
                bluff_prob = min(0.50, (fr - bluff_fr_thresh) * 1.5
                                 + pressure * 0.15)
                if random.random() < bluff_prob:
                    amt = self._raise_size(pot, 0.50, min_r, max_r)
                    self.logger.info(f"BLUFF s={street} eq={eq:.2f} fr={fr:.2f} prob={bluff_prob:.2f}")
                    self._pa_raised_this_street = True
                    return (AT.RAISE.value, amt, 0, 0)

        # ── Check / fold ──
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)

        if facing_raise:
            thresh = self._call_threshold(pot_odds, street, pressure)
            self.logger.info(f"FOLD s={street} eq={eq:.2f} thresh={thresh:.2f}")
            return (AT.FOLD.value, 0, 0, 0)

        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    # ── Main action ───────────────────────────────────────────────────────────

    def act(self, observation, reward, terminated, truncated, info):
        if terminated:
            return (AT.FOLD.value, 0, 0, 0)

        hand_number     = info.get("hand_number", 0)
        remaining_hands = max(1000 - hand_number, 0)

        # ── Autowin: fold once lead is mathematically insurmountable ──
        # Bleed = 3 chips/2 hands; threshold guarantees final bankroll sub-10.
        if self._cumulative >= (3 * remaining_hands + 1) // 2 + 2:
            if observation["street"] == 0:
                self.logger.info(f"AUTOWIN lead={self._cumulative:+.0f} remaining={remaining_hands}")
            return (AT.FOLD.value, 0, 0, 0)

        # ── Time budget detection (once per match) ──
        tl = float(observation.get("time_left", 500.0))
        if self._time_budget is None:
            self._time_budget = tl
            self.logger.info(f"TIME BUDGET detected: {self._time_budget:.0f}s")
        self._time_left = tl

        # ── Position: SB (blind_position=0) = IP post-flop in heads-up ──
        self._in_position = (observation.get("blind_position", 0) == 0)

        # ── Match pressure ──
        pressure = self._match_pressure(remaining_hands)

        street   = observation["street"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        board    = [c for c in observation["community_cards"] if c != -1]
        opp_disc = list(observation.get("opp_discarded_cards", [-1, -1, -1]))
        my_disc  = list(observation.get("my_discarded_cards", [-1, -1, -1]))
        valid    = observation["valid_actions"]

        # All known dead cards: opponent discards + our discards
        dead = [c for c in opp_disc if c != -1] + [c for c in my_disc if c != -1]

        # ── Opponent model update ──
        opp_last = str(observation.get("opp_last_action", ""))
        pot = observation["my_bet"] + observation["opp_bet"]
        if opp_last and self._prev_street >= 0:
            self._update_opp_model(self._prev_street, opp_last, pot)
        # Reset raise-war tracker on new street (v22 Fix 2)
        if street != self._prev_street:
            self._pa_raised_this_street = False
        self._prev_street = street

        # ── Track opponent discard tendencies (once per hand when discards first visible) ──
        opp_disc_cards = [c for c in opp_disc if c != -1]
        if len(opp_disc_cards) == 3 and not self._hand_discard_tracked:
            self._hand_discard_tracked = True
            self._opp_discard_total += 1
            for c in opp_disc_cards:
                self._opp_discard_suit_counts[suit(c)] += 1
                self._opp_discard_rank_counts[rank(c)] += 1

        self.logger.info(
            f"H{self._hand_count} s{street} tl={tl:.0f} IP={self._in_position} | "
            f"hole={[PokerEnv.int_card_to_str(c) for c in my_cards]} | "
            f"cum={self._cumulative:+.0f} press={pressure:+.2f} opp={self._opp_type()}"
        )

        # ── Discard ──
        if valid[AT.DISCARD.value]:
            (i, j), eq = self._best_discard(my_cards, board, dead)
            self.logger.info(f"DISCARD keep=({i},{j}) eq={eq:.2f}")
            return (AT.DISCARD.value, 0, i, j)

        # ── Pre-flop (5 cards) ──
        if street == 0 and len(my_cards) >= 5:
            return self._act_preflop(observation, my_cards, pressure)

        # ── Post-flop equity ──
        if len(my_cards) != 2:
            my_cards = my_cards[:2]

        opp_discards = [c for c in opp_disc if c != -1]
        if len(opp_discards) == 3:
            # Streets 2-3: we know opponent's 3 discards. Model their hand as
            # best-2-of-5 where 3 of the 5 are the known discards + 2 unknown.
            eq = self._equity_post_discard(my_cards, board, dead, opp_discards, n=self._n_samples(400))
        else:
            # Streets 0-1: opponent has 5 cards, pick best 2
            eq = self._equity(my_cards, board, dead, n=self._n_samples(400))

        return self._act_betting(observation, eq, pressure)

    # ── Showdown processing ────────────────────────────────────────────────────

    def _process_showdown(self, observation, info):
        """Record opponent's hand class at showdown, correlated with their actions."""
        try:
            # Determine which cards are opponent's
            my_cards_obs = [c for c in observation["my_cards"] if c != -1]
            my_strs = {PokerEnv.int_card_to_str(c) for c in my_cards_obs}
            p0_strs = set(info.get("player_0_cards", []))

            if my_strs & p0_strs:  # overlap → we're player 0
                opp_card_strs = info["player_1_cards"]
            else:
                opp_card_strs = info["player_0_cards"]

            board_strs = info.get("community_cards", [])
            if len(opp_card_strs) < 2 or len(board_strs) < 3:
                return

            # Evaluate opponent's hand class
            opp_treys = [Card.new(s) for s in opp_card_strs]
            board_treys = [Card.new(s) for s in board_strs]
            opp_rank = self.evaluator.evaluate(opp_treys, board_treys)
            opp_hc = self.evaluator.get_rank_class(opp_rank)

            # Record: for each street where opponent raised, log their hand class
            for s in self._hand_opp_raised_streets:
                self._opp_raise_showdown[s].append(opp_hc)

            # Record bet sizing correlation
            for frac in self._hand_opp_raise_fracs:
                self._opp_sizing_data.append((frac, opp_hc))

            # ── Track opponent's kept cards (what they prefer to hold) ──
            # Convert opponent card strings to ints for rank/suit analysis
            self._opp_showdown_count += 1
            opp_ints = []
            for cs in opp_card_strs:
                r_idx = PokerEnv.RANKS.index(cs[0])
                s_idx = PokerEnv.SUITS.index(cs[1])
                opp_ints.append(s_idx * NUM_RANKS + r_idx)

            if len(opp_ints) == 2:
                r0, r1 = rank(opp_ints[0]), rank(opp_ints[1])
                s0, s1 = suit(opp_ints[0]), suit(opp_ints[1])
                if r0 == r1:
                    self._opp_kept_paired += 1
                if s0 == s1:
                    self._opp_kept_suited += 1
                if abs(r0 - r1) <= 2:
                    self._opp_kept_connected += 1
                if r0 >= 6 and r1 >= 6:  # 8, 9, A
                    self._opp_kept_high += 1

            self.logger.info(
                f"SHOWDOWN-DATA opp={opp_card_strs} hc={opp_hc} "
                f"raised_streets={self._hand_opp_raised_streets} "
                f"keeps: paired={self._opp_kept_paired}/{self._opp_showdown_count} "
                f"suited={self._opp_kept_suited}/{self._opp_showdown_count} "
                f"connected={self._opp_kept_connected}/{self._opp_showdown_count}"
            )
        except Exception as e:
            self.logger.error(f"Showdown processing error: {e}")

    # ── Observation (end-of-hand accounting) ─────────────────────────────────

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated:
            self._hand_count += 1
            self._cumulative += reward
            self._prev_street = -1
            self._pa_raised_this_street = False   # reset raise-war tracker
            # Terminal folds by opponent appear here (not in act's model update)
            opp_term = str(observation.get("opp_last_action", ""))
            if opp_term == "FOLD":
                self._opp_fold_n   += 1
                self._opp_action_n += 1
            self.logger.info(
                f"{'WON' if reward > 0 else 'LOST'} {abs(reward):.0f} | "
                f"cum={self._cumulative:+.0f}"
            )

            # ── Showdown analysis: record opponent hand class vs their actions ──
            if "player_0_cards" in info:
                self._process_showdown(observation, info)

            # ── Reset per-hand tracking ──
            self._hand_opp_raised_streets = set()
            self._hand_opp_raise_fracs = []
            self._hand_discard_tracked = False
