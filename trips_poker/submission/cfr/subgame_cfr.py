# submission/cfr/subgame_cfr.py
"""
V4 Subgame CFR-D Solver — DeepStack-style continual re-solving.

Key changes from v3:
- Depth-limited CFR: at street boundaries, queries CounterfactualValueNet instead of recursing
- CFR-D gadget: augmented root for opponent range reconstruction (not binary threshold)
- Per-hand CFV interface: _cfr_iteration returns (351,) not scalar
- Linear time-weighting: strategy_sum += (t+1) * strategy
- Chance-node handling: update_for_new_street() filters r1, retrieves v2 from cache, rebuilds tree
- HAND_CONTAINS_CARD: precomputed (27,351) boolean mask at module load
"""
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

import numpy as np

from submission.card_utils import (
    NUM_CARDS, combo_index_2, combo_index_5, COMB
)
from submission.cfr.action_abstraction import (
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, RAISE_OVERBET,
    N_ACTIONS, ALL_RAISES, resolve_raise_amount,
)
from submission.cfr.utility import tournament_utility, tournament_utility_vec

# ── Precomputed constants ────────────────────────────────────────────────────

# All C(27,2) = 351 two-card kept hand combos, ordered by combo_index_2
# (combinatorial number system order, not lexicographic order)
_all_pairs = list(combinations(range(NUM_CARDS), 2))
ALL_HANDS = sorted(_all_pairs, key=lambda p: combo_index_2(p))
assert len(ALL_HANDS) == 351

# HAND_CONTAINS_CARD[c][h] = True iff hand h (index into ALL_HANDS) contains card c
# Shape: (27, 351) bool
HAND_CONTAINS_CARD = np.zeros((NUM_CARDS, len(ALL_HANDS)), dtype=bool)
for _h_idx, (_c1, _c2) in enumerate(ALL_HANDS):
    HAND_CONTAINS_CARD[_c1][_h_idx] = True
    HAND_CONTAINS_CARD[_c2][_h_idx] = True

HAND_TO_IDX = {h: i for i, h in enumerate(ALL_HANDS)}

# ── Value-net index permutation ────────────────────────────────────────────────
# Value nets were trained with ranges in LEXICOGRAPHIC order (itertools.combinations),
# but the solver uses COMBO_INDEX_2 order (combinatorial number system).
# These permutation arrays convert between the two orderings:
#   _COMBO_TO_LEX[combo_idx] = lex_idx   (solver → net input)
#   _LEX_TO_COMBO[lex_idx]   = combo_idx (net output → solver)
_lex_pairs = list(combinations(range(NUM_CARDS), 2))  # lexicographic order
_COMBO_TO_LEX = np.zeros(351, dtype=np.int32)
_LEX_TO_COMBO = np.zeros(351, dtype=np.int32)
for _lex_idx, _pair in enumerate(_lex_pairs):
    _combo_idx = HAND_TO_IDX[_pair]
    _COMBO_TO_LEX[_combo_idx] = _lex_idx
    _LEX_TO_COMBO[_lex_idx] = _combo_idx

# Lazy-loaded tables
_HAND_RANKS = None
_HAND_STRENGTH_2 = None
_OPTIMAL_KEEP_TABLE = None


def _get_tables():
    global _HAND_RANKS, _HAND_STRENGTH_2, _OPTIMAL_KEEP_TABLE
    if _HAND_RANKS is None:
        from submission.equity import (
            HAND_RANKS, HAND_STRENGTH_2, OPTIMAL_KEEP_TABLE,
            make_action_weights_fn
        )
        _HAND_RANKS = HAND_RANKS
        _HAND_STRENGTH_2 = HAND_STRENGTH_2
        _OPTIMAL_KEEP_TABLE = OPTIMAL_KEEP_TABLE
    return _HAND_RANKS, _HAND_STRENGTH_2, _OPTIMAL_KEEP_TABLE


# ── Backward compatibility helpers (used by player.py v3 and existing tests) ─

# Precompute C(n,k) for local_flop_index
_COMB_PY = [[0] * 8 for _ in range(28)]
for _n in range(28):
    _COMB_PY[_n][0] = 1
    for _k in range(1, min(_n + 1, 8)):
        _COMB_PY[_n][_k] = _COMB_PY[_n - 1][_k - 1] + _COMB_PY[_n - 1][_k]

KEEP_INDEX_TO_PAIR = [(i, j) for i in range(5) for j in range(i+1, 5)]
assert len(KEEP_INDEX_TO_PAIR) == 10


def local_flop_index(hand5: tuple, flop: tuple) -> int:
    remaining = sorted(c for c in range(NUM_CARDS) if c not in hand5)
    pos_lookup = {c: i for i, c in enumerate(remaining)}
    p0, p1, p2 = sorted(pos_lookup[c] for c in flop)
    return _COMB_PY[p0][1] + _COMB_PY[p1][2] + _COMB_PY[p2][3]


def expand_to_postdiscard_range(preflop_range: dict, flop: tuple) -> dict:
    _, _, opt_table = _get_tables()
    range_2card = defaultdict(float)
    for hand5, w in preflop_range.items():
        hand5_sorted = tuple(sorted(hand5))
        flop_idx = local_flop_index(hand5_sorted, flop)
        keep_pair_idx = int(opt_table[combo_index_5(hand5_sorted), flop_idx])
        i, j = KEEP_INDEX_TO_PAIR[keep_pair_idx]
        kept = (hand5_sorted[i], hand5_sorted[j])
        range_2card[kept] += w
    return dict(range_2card)


# ── V4 Helper Functions ──────────────────────────────────────────────────────

def build_our_range(our_5_cards: tuple, flop_3: tuple) -> np.ndarray:
    """
    Return point-mass range vector (351,) for our optimal kept 2-card hand.
    Uses OPTIMAL_KEEP_TABLE to pick the best 2 cards given flop.
    Fallback: uniform over non-dead hands.
    """
    r = np.zeros(351, dtype=np.float32)
    try:
        _, _, opt_table = _get_tables()
        hand5_sorted = tuple(sorted(our_5_cards))
        h5_idx = combo_index_5(hand5_sorted)
        flop_idx = local_flop_index(hand5_sorted, flop_3)
        keep_pair_idx = int(opt_table[h5_idx, flop_idx])
        i, j = KEEP_INDEX_TO_PAIR[keep_pair_idx]
        kept = tuple(sorted([hand5_sorted[i], hand5_sorted[j]]))
        hand_idx = HAND_TO_IDX.get(kept)
        if hand_idx is not None:
            r[hand_idx] = 1.0
            return r
    except Exception:
        pass
    # Fallback: uniform over hands not colliding with known cards
    dead = set(our_5_cards) | set(flop_3)
    for h_idx, (c1, c2) in enumerate(ALL_HANDS):
        if c1 not in dead and c2 not in dead:
            r[h_idx] = 1.0
    s = r.sum()
    if s > 0:
        r /= s
    return r


# ── Phantom range widening ────────────────────────────────────────────────
# Widened P1 range: prevents P2 from perfectly adapting to a known single
# hand. Adds phantom nut/air hands so P2 faces genuine uncertainty.
# Adaptive count by street: flop=2, turn=4, river=6 phantoms.

_PHANTOM_CONFIG = {
    3: (1, 0.70),   # flop:  1 nut + 1 air, 70% our hand
    2: (1, 0.70),   # turn:  1 nut + 1 air, 70% our hand
    1: (1, 0.70),   # river: 1 nut + 1 air, 70% our hand
}


def build_opponent_range(our_5_cards: tuple, flop_3: tuple,
                         opp_discards: tuple = None,
                         hand_actions: list = None,
                         opp_profile=None,
                         preflop_weights=None) -> np.ndarray:
    """
    Build opponent's post-discard range using discard-weighted prior + Bayesian narrowing.

    V10 FIX: Replaces binary rational filter with soft probability weighting.
    The discard mechanic creates a 680x non-uniform distribution over opponent
    hands — ignoring this made the solver assume a near-uniform range, causing
    systematic over-calling (DeepStack: "counterfactual values depend on how
    players play to reach the public state, i.e., the players' ranges").

    Pipeline:
      1. Compute discard-weighted prior: P(opponent keeps hand h | our 5 cards, flop)
         by enumerating all C(19,5)=11,628 possible opponent deals and looking up
         optimal keep from precomputed table. ~5ms.
      2. If opp_discards known: zero out hands inconsistent with known discards.
      3. Apply Bayesian action-sequence narrowing on top of prior.
      4. Zero out hands containing depleted-rank cards (3-suit game constraint).
      5. Normalize.
    """
    from submission.compute_range_prior import compute_opponent_prior

    dead = set(our_5_cards) | set(flop_3)
    known_cards = set(our_5_cards) | set(flop_3)
    if opp_discards is not None:
        known_cards |= set(opp_discards)

    # Depleted ranks: if all 3 suits of a rank are known, opponent can't have it
    rank_counts = {}
    for c in known_cards:
        if c == -1: continue
        r_val = c % 9
        rank_counts[r_val] = rank_counts.get(r_val, 0) + 1
    depleted_ranks = {r for r, count in rank_counts.items() if count >= 3}

    # Step 1: Discard-weighted prior (~5ms, vectorized numpy)
    # If preflop_weights provided, use weighted version that incorporates
    # P(opponent preflop actions | hand) into the prior
    if preflop_weights is not None:
        from submission.compute_range_prior import compute_opponent_prior_weighted
        r = compute_opponent_prior_weighted(our_5_cards, flop_3, preflop_weights).astype(np.float32)
    else:
        r = compute_opponent_prior(our_5_cards, flop_3).astype(np.float32)

    # Step 2: If opponent discards known, zero out inconsistent hands
    if opp_discards is not None and all(c != -1 for c in opp_discards):
        opp_disc_set = set(opp_discards)
        for c in opp_disc_set:
            if 0 <= c < NUM_CARDS:
                r[HAND_CONTAINS_CARD[c]] = 0.0
        # Also zero hands that include any dead card (our cards, board)
        for c in dead:
            r[HAND_CONTAINS_CARD[c]] = 0.0
    else:
        # Zero out hands colliding with our known cards
        for c in dead:
            r[HAND_CONTAINS_CARD[c]] = 0.0

    # Step 3: Zero out hands with depleted-rank cards
    if depleted_ranks:
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            if (c1 % 9) in depleted_ranks or (c2 % 9) in depleted_ranks:
                r[h_idx] = 0.0

    # Step 4: Apply Bayesian action-sequence narrowing
    if hand_actions and opp_profile:
        from submission.equity import make_action_weights_fn
        # Get candidate hands (nonzero in prior)
        nonzero_idx = np.nonzero(r)[0]
        if len(nonzero_idx) > 0:
            candidate_hands = [ALL_HANDS[i] for i in nonzero_idx]
            weights_fn = opp_profile.opp_weights_fn if opp_profile else (lambda h: 1.0)
            weights_fn = make_action_weights_fn(
                weights_fn, candidate_hands, list(flop_3), hand_actions)
            for i in nonzero_idx:
                r[i] *= float(weights_fn(ALL_HANDS[i]))

    # Step 5: Normalize
    s = r.sum()
    if s > 0:
        r /= s
    else:
        # Safety fallback: uniform over non-colliding hands
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            if c1 not in dead and c2 not in dead:
                r[h_idx] = 1.0
        s = r.sum()
        if s > 0:
            r /= s
    return r


def compute_uniform_cfv(flop_3: tuple, match_margin: float,
                         hands_remaining: int) -> np.ndarray:
    """
    HS2-based expected value against uniform opponent range.
    Returns (351,) float32 in CHIP SCALE (matching subgame CFV scale).

    Previous bug: used tournament_utility (~0.002 scale) while subgame CFVs
    are in chip scale (~5.0). The CFR-D gadget computes advantage = cfv - v2,
    so scales must match.
    """
    _, hs2, _ = _get_tables()
    pot_proxy = 10.0  # assumed average pot for entering subgame
    mean_hs2 = float(hs2.mean())
    # Chip-scale CFV: (hand_strength - avg_strength) * pot / 2
    cfv = ((hs2 - mean_hs2) * pot_proxy / 2).astype(np.float32)
    return cfv


def compute_profile_cfv(opp_profile, flop_3: tuple,
                         match_margin: float, hands_remaining: int) -> np.ndarray:
    """
    Weight uniform CFV by opponent aggression tendency per hand-strength bucket.
    Falls back to compute_uniform_cfv if profile has no data.
    Returns (351,) float32 in chip scale.
    """
    base = compute_uniform_cfv(flop_3, match_margin, hands_remaining)
    try:
        _, hs2, _ = _get_tables()
        weights = np.array([
            1.0 + float(opp_profile.aggression_by_strength(float(hs2[i])))
            for i in range(351)
        ], dtype=np.float32)
        return base * weights
    except Exception:
        return base


# ── SubgameNode ──────────────────────────────────────────────────────────────

@dataclass
class SubgameNode:
    player: int
    pot: float
    stack: float
    ip_in_pot: float
    oop_in_pot: float
    action_history: tuple
    streets_remaining: int  # V4: streets left after this node's betting ends
    regret_sum: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    children: dict = field(default_factory=dict)
    # Populated by _precompute_depth_limit:
    _cached_p1_cfv: np.ndarray = field(default=None, init=False)
    _cached_p2_cfv: np.ndarray = field(default=None, init=False)

    def get_strategy(self) -> np.ndarray:
        """Current mixed strategy via regret-matching. Returns (N_ACTIONS,)."""
        valid = list(self.children.keys())
        if not valid:
            return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
        pos = np.maximum(self.regret_sum, 0.0)
        total = pos[valid].sum()
        s = np.zeros(N_ACTIONS, dtype=np.float32)
        if total > 0:
            for a in valid:
                s[a] = pos[a] / total
        else:
            for a in valid:
                s[a] = 1.0 / len(valid)
        return s

    def get_average_strategy(self) -> np.ndarray:
        """Average strategy from strategy_sum. Returns (N_ACTIONS,)."""
        valid = list(self.children.keys())
        if not valid:
            return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
        total = self.strategy_sum[valid].sum()
        if total > 0:
            s = self.strategy_sum / total
        else:
            s = np.zeros(N_ACTIONS, dtype=np.float32)
            for a in valid:
                s[a] = 1.0 / len(valid)
        return s.astype(np.float32)


# ── SubgameSolver ────────────────────────────────────────────────────────────

RAISE_CAP = 3  # max raises per street (3 = open-raise + 3bet + 4bet)


class SubgameSolver:
    """
    V4 DeepStack-style continual re-solving subgame CFR-D solver.

    Lifecycle per hand:
      1. initialize(our_5_cards, flop_3, pot, stack, opp_profile, ...) — at discard time
      2. observe_action(abstract_action) — after each real game action
      3. update_for_new_street(card) — when turn or river card is revealed
      4. get_action_cfvs() — at each decision point for action selection
    """

    def __init__(self):
        self._initialized = False
        self._iterations_run = 0
        self._current_root = None
        self._root = None
        self._last_cfvs = np.zeros(N_ACTIONS, dtype=np.float32)
        self._hand_cfvs = np.zeros(N_ACTIONS, dtype=np.float32)  # pass-2 hand-specific CFVs
        self._r1 = np.ones(351, dtype=np.float32) / 351.0
        self._v2 = np.zeros(351, dtype=np.float32)
        self._current_opp_range = np.ones(351, dtype=np.float32) / 351.0
        self._last_opp_subgame_cfv = np.zeros(351, dtype=np.float32)
        self._per_card_v2_cache = {}
        self._board_equity_cache = {}
        self._known_cards = set()
        self._dead_cards = set()  # only board cards (for equity computation)
        self._current_pot = 0.0
        self._current_stack = 200.0
        self._streets_remaining = 3
        self._budget_per_street = 0.3
        self._budget_end = 0.0
        self._value_net_turn = None
        self._value_net_flop = None
        self._value_net_river = None
        self._match_margin = 0.0
        self._hands_remaining = 500
        self._opp_fold_rate = 0.30
        self._opp_raise_rate = 0.30
        self._opp_vpip = 1.0  # default: assume opponent plays all hands
        self._opp_confidence = 0.0  # min(hands_observed / 50, 1.0)
        self._rnr_p = 0.0  # RNR blend parameter: prob of locked branch
        self._exploit_alpha = 0.0
        self._in_position = False


    def get_our_rhs(self) -> float:
        """Return our current hand's Relative Hand Strength (0.0 - 1.0)."""
        hand_ranks, _, _ = _get_tables()
        if hand_ranks is None or len(self._board_cards) < 3:
            return 0.5
        
        from submission.card_utils import combo_index_7
        board = tuple(self._board_cards)
        n_cards = len(board)
        # Use full 5-card board for evaluation (fill with -1 if needed, though subgame usually has 3+)
        board_5 = tuple(list(board) + [-1]*(5-n_cards))
        
        our_h1 = self._our_hand_idx
        if our_h1 < 0: return 0.5
        
        # Calculate score for our hand
        our_key = tuple(sorted(ALL_HANDS[our_h1] + board_5))
        our_score = float(hand_ranks[combo_index_7(our_key)])
        
        # Compare vs all physically possible hands
        dead = self._dead_cards
        wins = 0
        total = 0
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            if c1 not in dead and c2 not in dead:
                key = tuple(sorted((c1, c2) + board_5))
                score = float(hand_ranks[combo_index_7(key)])
                if score > our_score: wins += 1
                total += 1
        return wins / max(1, total - 1)

    def get_opp_class_distribution(self) -> dict:
        """Summarize the converged opponent range by hand class."""
        hand_ranks, _, _ = _get_tables()
        if hand_ranks is None or len(self._board_cards) < 3:
            return {}
            
        from submission.card_utils import combo_index_7
        board = tuple(self._board_cards)
        n_cards = len(board)
        board_5 = tuple(list(board) + [-1]*(5-n_cards))
        
        from gym_env import PokerEnv
        evaluator = PokerEnv().evaluator
        
        dist = {}
        r = self._current_opp_range
        dead = self._dead_cards
        
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            prob = float(r[h_idx])
            if prob < 0.001: continue
            
            key = tuple(sorted((c1, c2) + board_5))
            score = int(hand_ranks[combo_index_7(key)])
            class_idx = evaluator.get_rank_class(score)
            class_str = evaluator.class_to_string(class_idx)
            
            dist[class_str] = dist.get(class_str, 0.0) + prob
            
        return {k: round(v, 3) for k, v in sorted(dist.items(), key=lambda x: -x[1])}

    def _apply_continuous_range_weighting(self) -> None:
        """Continuously weight opponent range based on observed selectivity.

        Fix #19: Two signals for opponent selectivity:
        1. fold_rate (fold rate to our raises) — how often they fold postflop
        2. VPIP (voluntarily put money in pot) — how selective preflop

        Low VPIP is the strongest signal: an opponent with VPIP=0.12 only
        plays 12% of hands, so their range when they DO play is top-12%.
        V8 logs showed opponents with 86-88% prefold rate (VPIP ~12-14%)
        but the old weighting only used fold_rate, which was often low
        for these opponents (they're stations who call — low fold rate
        but very narrow range).

        All scaling is continuous — no thresholds or classifications.
        """
        fold_rate = self._opp_fold_rate
        vpip = self._opp_vpip
        confidence = self._opp_confidence

        # Signal 1: fold rate (original) — opponents who fold more hold stronger
        # Blend scales linearly: 0 at 0.30, up to 0.60 at 0.80+
        fold_blend = max(0.0, (fold_rate - 0.30)) * 1.2
        fold_blend = min(0.60, fold_blend)

        # Signal 2: VPIP selectivity — low VPIP = very strong range
        # VPIP 1.0 (plays everything) → 0 blend
        # VPIP 0.50 → moderate blend ~0.30
        # VPIP 0.15 → strong blend ~0.51
        # VPIP 0.05 → very strong blend ~0.57
        vpip_blend = max(0.0, (1.0 - vpip) * 0.60)

        # Use the stronger signal
        blend = max(fold_blend, vpip_blend)
        blend *= confidence  # scale by observation confidence

        if blend < 0.01:
            return  # negligible effect, skip computation

        cards_to_come = 5 - len(self._board_cards)
        equity = self._compute_board_equity(cards_to_come)
        if equity is None:
            return

        opp = self._current_opp_range
        nz = opp > 0
        if not np.any(nz):
            return

        # Weight by HS² — strong hands get quadratically more weight
        weights = np.where(nz, equity ** 2, 0.0).astype(np.float32)
        opp_new = blend * (opp * weights) + (1.0 - blend) * opp
        s = opp_new.sum()
        if s > 0:
            self._current_opp_range = opp_new / s

    def _build_widened_range(self, equity: np.ndarray, streets_remaining: int) -> np.ndarray:
        """Build widened P1 range with phantom nut/air hands.

        Uses per-hand equity to select the strongest (nut) and weakest (air)
        hands as phantoms. P2 faces genuine range uncertainty, preventing the
        degenerate Nash equilibrium where P2 perfectly adapts to a known hand.

        Args:
            equity: (351,) per-hand equity vs uniform opponent on current board
            streets_remaining: 3=flop, 2=turn, 1=river (determines phantom count)
        Returns:
            (351,) float32 normalized range vector
        """
        our_h1 = self._our_hand_idx
        if our_h1 < 0 or equity is None:
            return self._r1.copy()

        n_each, our_weight = _PHANTOM_CONFIG.get(streets_remaining, (1, 0.70))
        total_phantom = 1.0 - our_weight

        # Continuous air weighting: linear function of opponent fold rate.
        # Low fold rate (station) → more air → solver pot-controls.
        # High fold rate (nit) → more nuts → solver value-bets aggressively.
        # No thresholds — smooth function scaled by confidence.
        #   fold_rate=0.0 → air=0.75, fold_rate=0.50 → air=0.50, fold_rate=1.0 → air=0.25
        opp_fold = getattr(self, '_opp_fold_rate', 0.50)
        confidence = getattr(self, '_opp_confidence', 0.0)
        base_air = 0.75 - 0.50 * opp_fold  # linear: 0.75 at 0, 0.25 at 1.0
        air_frac = 0.50 + (base_air - 0.50) * confidence  # blend toward 0.50 with low confidence
        air_frac = max(0.25, min(0.75, air_frac))

        air_weight = total_phantom * air_frac
        nut_weight = total_phantom * (1.0 - air_frac)

        # Valid phantom candidates: no dead-card conflicts, not our actual hand
        valid_for_phantom = np.ones(351, dtype=bool)
        for c in self._dead_cards:
            valid_for_phantom &= ~HAND_CONTAINS_CARD[c]
        valid_for_phantom[our_h1] = False

        valid_indices = np.where(valid_for_phantom)[0]
        if len(valid_indices) < 2 * n_each:
            return self._r1.copy()  # not enough candidates

        # Rank by equity; pick highest as nuts, lowest as air
        eq_valid = equity[valid_indices]
        sorted_order = np.argsort(eq_valid)  # ascending
        air_idx = valid_indices[sorted_order[:n_each]]
        nut_idx = valid_indices[sorted_order[-n_each:]]

        # Build weighted range
        r1 = np.zeros(351, dtype=np.float32)
        r1[our_h1] = our_weight
        per_nut = nut_weight / max(1, len(nut_idx))
        for h in nut_idx:
            r1[h] = per_nut
        per_air = air_weight / max(1, len(air_idx))
        for h in air_idx:
            r1[h] = per_air

        s = r1.sum()
        if s > 0:
            r1 /= s
        return r1

    def initialize(self, our_5_cards: tuple, flop_3: tuple,
                   pot: float, stack: float,
                   opp_profile, match_margin: float, hands_remaining: int,
                   budget_seconds: float,
                   value_net_flop, value_net_turn,
                   value_net_river=None,
                   in_position: bool = False,
                   opp_discards: tuple = None,
                   hand_actions: list = None,
                   preflop_weights=None,
                   our_kept: tuple = None) -> None:
        self._value_net_flop = value_net_flop
        self._value_net_turn = value_net_turn
        self._value_net_river = value_net_river
        self._match_margin = match_margin
        self._hands_remaining = hands_remaining
        self._per_card_v2_cache = {}
        self._board_equity_cache = {}
        self._board_cards = list(flop_3)  # grows: flop→turn→river
        # r1: point mass on our actual kept hand (from player's discard decision)
        if our_kept is not None:
            kept_sorted = tuple(sorted(our_kept))
            hand_idx = HAND_TO_IDX.get(kept_sorted)
            if hand_idx is not None:
                self._r1 = np.zeros(351, dtype=np.float32)
                self._r1[hand_idx] = 1.0
            else:
                self._r1 = build_our_range(our_5_cards, flop_3)
        else:
            self._r1 = build_our_range(our_5_cards, flop_3)
        # Fast scalar index for our hand (r1 is almost always a point mass)
        nz = np.nonzero(self._r1)[0]
        self._our_hand_idx = int(nz[0]) if len(nz) == 1 else -1

        # Dead cards: all dealt cards are permanently out of play.
        # Community cards are pre-dealt at game start; discards never return.
        # Dead = our 5 original cards + board + opponent's 5 (unknown but dealt).
        # We know: our 5 cards + flop 3. Opponent's 5 are unknown.
        self._dead_cards = set(our_5_cards) | set(flop_3)
        self._known_cards = set(our_5_cards) | set(flop_3)

        # Widen P1 range with phantom nut/air hands.
        # Compute board equity for phantom selection, then build widened range.
        cards_to_come = 5 - len(self._board_cards)  # flop: 2
        _phantom_eq = self._compute_board_equity(cards_to_come)
        if _phantom_eq is not None:
            self._r1 = self._build_widened_range(_phantom_eq, 3)

        # v2: confidence-blended CFV for opponent
        alpha = min(opp_profile.hands_observed / 150.0, 0.65)
        v2_profile = compute_profile_cfv(opp_profile, flop_3, match_margin, hands_remaining)
        v2_uniform = compute_uniform_cfv(flop_3, match_margin, hands_remaining)
        self._v2 = (alpha * v2_profile + (1.0 - alpha) * v2_uniform).astype(np.float32)

        self._current_pot = pot
        self._current_stack = stack
        self._streets_remaining = 3
        self._in_position = in_position

        # Opponent tendencies — continuous values, no classifications.
        # These feed into phantom range composition and opponent range
        # weighting as smooth functions, not binary switches.
        try:
            opp_ctx = opp_profile.get_context(1)
            # Use postflop fold rate when available (avoids preflop pollution).
            # Fall back to aggregate if no postflop data yet.
            if opp_ctx.get('postflop_raises_faced', 0) >= 3:
                raw_fold = float(opp_ctx.get('postflop_fold_rate', 0.30))
            else:
                raw_fold = float(opp_ctx.get('fold_rate_to_our_raise', 0.30))
            self._opp_fold_rate = max(0.05, min(0.95, raw_fold))
            self._opp_raise_rate = max(0.05, min(0.95,
                float(opp_ctx.get('raise_rate', 0.30))))
            # Fix #19: Track opponent VPIP (Voluntarily Put $ In Pot).
            # Low VPIP = very selective opponent = strong range when they play.
            # V8 logs: opponents prefold 86-88%, VPIP ~12-14%.
            self._opp_vpip = float(opp_ctx.get('vpip', 1.0))
        except Exception:
            self._opp_fold_rate = 0.30
            self._opp_raise_rate = 0.30
            self._opp_vpip = 1.0
        self._opp_confidence = min(opp_profile.hands_observed / 50.0, 1.0)
        self._exploit_alpha = 0.0
        self._opp_profile = opp_profile  # for raise size distribution

        # RNR parameter: probability of the "locked" branch where P2
        # plays the observed model strategy.  Scales with both deviation
        # from Nash-like fold rate (0.30) and observation confidence.
        # p=0 → pure Nash.  p=0.50 → max exploitation (capped for safety).
        deviation = abs(self._opp_fold_rate - 0.30)
        self._rnr_p = min(0.50, deviation * self._opp_confidence)

        # Position fix: if we're IP (SB), opponent (BB) acts first postflop
        root_player = 1 if in_position else 0
        self._root = self._build_tree(
            player=root_player, pot=pot, stack=stack,
            ip_in_pot=pot / 2, oop_in_pot=pot / 2,
            action_history=(), raises_this_street=0,
            streets_remaining=3,
        )
        self._current_root = self._root

        # Flop needs enough for initial solve + observe_action rebuilds.
        # Old 20/30/50 left only ~1s for flop (consumed by initial solve,
        # nothing for observe_action). 35/30/35 gives flop ~2.5s: ~1.5s
        # initial + ~1s for observe_action subtree re-solves.
        self._budget_flop = budget_seconds * 0.35
        self._budget_turn = budget_seconds * 0.30
        self._budget_river = budget_seconds * 0.35
        self._budget_per_street = self._budget_flop  # Initial: flop
        self._current_opp_range = build_opponent_range(
            our_5_cards, flop_3,
            opp_discards=opp_discards,
            hand_actions=hand_actions,
            opp_profile=opp_profile,
            preflop_weights=preflop_weights,
        )

        # Continuous opponent range weighting: shift toward strong hands
        # proportional to fold rate and confidence.  No thresholds.
        self._apply_continuous_range_weighting()

        self._last_opp_subgame_cfv = self._v2.copy()

        self._precompute_depth_limit(self._root)
        self._prepare_flat_traversal(self._root)

        self._initialized = True
        self._iterations_run = 0
        self._last_cfvs[:] = 0.0  # Reset stale CFVs from previous hand
        self._hand_cfvs[:] = 0.0
        self._budget_end = time.time() + self._budget_per_street
        self._iterate_cfrd()
        self._extract_hand_specific_cfvs()  # pass 2: hand-specific CFVs

    def _build_tree(self, player: int, pot: float, stack: float,
                    ip_in_pot: float, oop_in_pot: float,
                    action_history: tuple, raises_this_street: int,
                    streets_remaining: int) -> SubgameNode:
        """Build betting tree for one street."""
        node = SubgameNode(
            player=player, pot=pot, stack=stack,
            ip_in_pot=ip_in_pot, oop_in_pot=oop_in_pot,
            action_history=action_history,
            streets_remaining=streets_remaining,
        )
        if action_history and action_history[-1] == FOLD:
            return node  # fold terminal

        # A CALL after a raise (or check-check) ends the street — terminal.
        # But we need to distinguish:
        # - "street ended normally" → depth-limited leaf (streets_remaining > 0)
        # - "both all-in after call" → showdown terminal (streets_remaining = 0)
        if len(action_history) >= 2 and action_history[-1] == CALL:
            # Street ended. If both players are all-in (stack=0), mark as
            # showdown terminal (sr=0) so it gets exact/proxy evaluation.
            if stack <= 0:
                node.streets_remaining = 0
            return node

        # Opening check followed by a call is handled below (empty history case).
        # If this is the first action on a new street with stack=0, both are
        # all-in from a previous street — this is a showdown terminal.
        if stack <= 0 and not action_history:
            node.streets_remaining = 0
            return node

        # FOLD only when facing a bet (when checking is free, fold is dominated)
        facing_bet = abs(ip_in_pot - oop_in_pot) > 0.01
        valid_actions = [FOLD, CALL] if facing_bet else [CALL]
        if raises_this_street < RAISE_CAP and stack > 0:
            valid_actions += [RAISE_SMALL, RAISE_LARGE, RAISE_OVERBET, RAISE_ALLIN]

        for a in valid_actions:
            if a == FOLD:
                child = SubgameNode(
                    player=1 - player, pot=pot, stack=stack,
                    ip_in_pot=ip_in_pot, oop_in_pot=oop_in_pot,
                    action_history=action_history + (FOLD,),
                    streets_remaining=streets_remaining,
                )
                node.children[a] = child
            elif a == CALL:
                call_amount = abs(ip_in_pot - oop_in_pot)
                new_pot = pot + call_amount
                new_stack = max(0.0, stack - call_amount)
                new_ip = max(ip_in_pot, oop_in_pot)
                new_oop = max(ip_in_pot, oop_in_pot)
                new_history = action_history + (CALL,)
                # Opening check (empty action_history) → opponent still gets to act.
                if not action_history:
                    child = self._build_tree(
                        player=1 - player,
                        pot=new_pot, stack=new_stack,
                        ip_in_pot=new_ip, oop_in_pot=new_oop,
                        action_history=new_history,
                        raises_this_street=raises_this_street,
                        streets_remaining=streets_remaining,
                    )
                else:
                    # CALL after a raise → street ends.
                    # If all-in (stack=0), mark as showdown (sr=0).
                    # Otherwise, depth-limited boundary.
                    new_sr = 0 if new_stack <= 0 else max(0, streets_remaining - 1)
                    child = SubgameNode(
                        player=1 - player, pot=new_pot, stack=new_stack,
                        ip_in_pot=new_ip, oop_in_pot=new_oop,
                        action_history=new_history,
                        streets_remaining=new_sr,
                    )
                node.children[a] = child
            else:
                current_bet = abs(ip_in_pot - oop_in_pot)
                raise_amt = resolve_raise_amount(a, pot, stack,
                                                  current_bet=current_bet)
                if raise_amt <= 0:
                    continue
                new_pot = pot + raise_amt
                new_stack = max(0.0, stack - raise_amt)
                if player == 0:
                    new_ip = ip_in_pot + raise_amt
                    new_oop = oop_in_pot
                else:
                    new_ip = ip_in_pot
                    new_oop = oop_in_pot + raise_amt
                # After a raise, opponent ALWAYS gets to respond (fold/call/re-raise)
                child = self._build_tree(
                    player=1 - player,
                    pot=new_pot, stack=new_stack,
                    ip_in_pot=new_ip, oop_in_pot=new_oop,
                    action_history=action_history + (a,),
                    raises_this_street=raises_this_street + 1,
                    streets_remaining=streets_remaining,
                )
                node.children[a] = child
        return node

    def _is_street_boundary(self, node: SubgameNode) -> bool:
        """True if node is a depth-boundary leaf (street ended, not fold)."""
        is_leaf = not node.children
        if not is_leaf:
            return False
        is_fold = (len(node.action_history) > 0 and
                   node.action_history[-1] == FOLD)
        return not is_fold and node.streets_remaining in (1, 2)

    def _precompute_depth_limit(self, node: SubgameNode) -> None:
        """Walk tree once; cache network values at ALL leaves (including folds)."""
        if not node.children:
            is_fold = (len(node.action_history) > 0 and
                       node.action_history[-1] == FOLD)
            if is_fold:
                # Cache fold terminal CFVs normalized to MWP-delta scale
                folder = 1 - node.player
                if folder == 0:
                    chip_p1 = np.full(351, -node.ip_in_pot, dtype=np.float32)
                    chip_p2 = np.full(351, +node.ip_in_pot, dtype=np.float32)
                    node._cached_p1_cfv = tournament_utility_vec(chip_p1, self._match_margin, self._hands_remaining)
                    node._cached_p2_cfv = tournament_utility_vec(chip_p2, -self._match_margin, self._hands_remaining)
                else:
                    chip_p1 = np.full(351, +node.oop_in_pot, dtype=np.float32)
                    chip_p2 = np.full(351, -node.oop_in_pot, dtype=np.float32)
                    node._cached_p1_cfv = tournament_utility_vec(chip_p1, self._match_margin, self._hands_remaining)
                    node._cached_p2_cfv = tournament_utility_vec(chip_p2, -self._match_margin, self._hands_remaining)
                return
            if self._is_street_boundary(node):
                # River net is more accurate at turn→river boundary (trained on river data).
                # Fall back to turn net, then to proxy.
                if node.streets_remaining == 2:
                    # Enumerating turn cards → query turn net (trained on 4-card boards)
                    net = self._value_net_turn
                elif node.streets_remaining == 1:
                    net = self._value_net_river if self._value_net_river is not None else self._value_net_turn
                else:
                    net = None
                if net is not None:
                    p1_cfv, p2_cfv = self._query_net_with_card_enumeration(node, net)
                else:
                    p1_cfv, p2_cfv = self._board_aware_proxy_cfv(node)
                node._cached_p1_cfv = p1_cfv
                node._cached_p2_cfv = p2_cfv
            elif node.streets_remaining == 0:
                # All-in showdown. Use exact evaluation if board is complete,
                # otherwise board-aware equity for remaining cards.
                cards_on_board = len(self._board_cards)
                if cards_on_board >= 5:
                    p1_cfv, p2_cfv = self._exact_river_cfv(node)
                else:
                    # All-in before river — enumerate remaining board cards
                    cards_to_come = 5 - cards_on_board
                    equity = self._compute_board_equity(cards_to_come)
                    if equity is not None:
                        chips = node.pot / 2
                        # P2: per-hand outcome vs our hand
                        p2_vs_us = self._board_equity_cache.get(
                            ('p2_vs_us', cards_to_come))
                        p2_chip = (p2_vs_us * chips).astype(np.float32) if p2_vs_us is not None else (-equity * chips).astype(np.float32)
                        p2_cfv = tournament_utility_vec(p2_chip, -self._match_margin, self._hands_remaining)
                        # P1: per-hand values for ALL alive hands
                        # (needed for per-hand P1 strategies in CFR iteration)
                        p1_chip = (equity * chips).astype(np.float32)
                        p1_cfv = tournament_utility_vec(p1_chip, self._match_margin, self._hands_remaining)
                    else:
                        # _hs2_proxy_cfv already uses tournament_utility_vec — do NOT wrap again
                        p1_cfv, p2_cfv = self._hs2_proxy_cfv(node)
                node._cached_p1_cfv = p1_cfv
                node._cached_p2_cfv = p2_cfv
            return
        for child in node.children.values():
            self._precompute_depth_limit(child)

    def _prepare_flat_traversal(self, root: SubgameNode) -> None:
        """Pre-compute flat post-order node list and per-node child arrays.

        Converts the recursive tree into flat arrays for fast iterative CFR.
        After calling this, _iterate_cfrd uses the flat iteration loop.
        """
        # Post-order DFS
        nodes = []
        node_to_idx = {}
        stack = [(root, False)]
        while stack:
            nd, processed = stack.pop()
            if processed or not nd.children:
                node_to_idx[id(nd)] = len(nodes)
                nodes.append(nd)
            else:
                stack.append((nd, True))
                for a in reversed(sorted(nd.children.keys())):
                    stack.append((nd.children[a], False))

        N = len(nodes)
        self._flat_nodes = nodes
        self._flat_N = N
        self._flat_root_idx = node_to_idx[id(root)]
        self._flat_node_map = node_to_idx  # node id → flat index (for subtree extraction)

        # Per-node arrays
        self._flat_is_terminal = [True] * N
        self._flat_is_fold_terminal = [False] * N  # fold vs showdown distinction
        self._flat_player = [0] * N
        self._flat_actions = [None] * N      # tuple of action indices
        self._flat_child_idx = [None] * N    # tuple of flat indices
        self._flat_n_children = [0] * N
        self._flat_regret_sum = [None] * N   # reference to node.regret_sum
        self._flat_strategy_sum = [None] * N
        
        self._flat_p2_terminal = [None] * N

        for i, nd in enumerate(nodes):
            self._flat_player[i] = nd.player
            self._flat_regret_sum[i] = nd.regret_sum
            self._flat_strategy_sum[i] = nd.strategy_sum
            if nd.children:
                self._flat_is_terminal[i] = False
                actions = tuple(sorted(nd.children.keys()))
                self._flat_actions[i] = actions
                self._flat_child_idx[i] = tuple(
                    node_to_idx[id(nd.children[a])] for a in actions)
                self._flat_n_children[i] = len(actions)
            else:
                is_fold = (len(nd.action_history) > 0 and
                           nd.action_history[-1] == FOLD)
                self._flat_is_fold_terminal[i] = is_fold
                self._flat_p2_terminal[i] = nd._cached_p2_cfv

        # Pre-allocate iteration buffers
        self._flat_p2_all = np.zeros((N, 351), dtype=np.float32)  # 2D pre-allocated
        # Copy terminal P2 values into the 2D array
        for i in range(N):
            if self._flat_is_terminal[i] and self._flat_p2_terminal[i] is not None:
                self._flat_p2_all[i] = self._flat_p2_terminal[i]
        self._flat_reach_p1 = np.zeros((N, 351), dtype=np.float64)  # per-hand P1 reach
        self._flat_reach_p2 = np.zeros((N, 351), dtype=np.float64)
        # Per-node strategy values (max 5 children)
        self._flat_sv = [None] * N
        self._flat_p2_scratch = np.empty(351, dtype=np.float32)  # scratch buffer
        # Per-hand P1 value buffer
        self._flat_p1_all = np.zeros((N, 351), dtype=np.float32)

        # Per-hand P2 strategies: opponent needs hand-dependent regrets/strategies
        # Without this, all 351 opponent hands play identical mixed strategies,
        # causing degenerate behavior (e.g., 100% RAISE_ALLIN with everything).
        self._flat_is_p2_decision = [False] * N
        self._flat_p2_regret = [None] * N       # (351, N_ACTIONS) per P2 decision node
        self._flat_p2_strat_sum = [None] * N    # (351, N_ACTIONS)
        self._flat_p2_strat_ph = [None] * N     # (351, N_ACTIONS) scratch per iteration
        for i, nd in enumerate(nodes):
            if nd.player == 1 and nd.children:
                self._flat_is_p2_decision[i] = True
                self._flat_p2_regret[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)
                self._flat_p2_strat_sum[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)
                self._flat_p2_strat_ph[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)

        # Per-hand P1 strategies: P1 needs hand-dependent regrets/strategies
        # so phantom nut hands raise and air hands fold, giving P2 realistic signals.
        self._flat_is_p1_decision = [False] * N
        self._flat_p1_regret = [None] * N       # (351, N_ACTIONS) per P1 decision node
        self._flat_p1_strat_sum = [None] * N    # (351, N_ACTIONS)
        self._flat_p1_strat_ph = [None] * N     # (351, N_ACTIONS) scratch per iteration
        for i, nd in enumerate(nodes):
            if nd.player == 0 and nd.children:
                self._flat_is_p1_decision[i] = True
                self._flat_p1_regret[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)
                self._flat_p1_strat_sum[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)
                self._flat_p1_strat_ph[i] = np.zeros((351, N_ACTIONS), dtype=np.float32)
                

        # RNR: pre-compute model strategies for P2 decision nodes.
        # Each entry is a (n_actions,) array aligned with the node's action tuple.
        # Used on the "locked" branch of the RNR chance node.
        self._flat_p2_model_sv = [None] * N
        rnr_p = self._rnr_p
        if rnr_p > 0.01:
            for i, nd in enumerate(nodes):
                if nd.player == 1 and nd.children:
                    actions = tuple(sorted(nd.children.keys()))
                    facing = (len(nd.action_history) > 0 and
                              nd.action_history[-1] >= RAISE_SMALL)
                    sv = self._compute_opp_model_sv(actions, facing)
                    self._flat_p2_model_sv[i] = np.array(sv, dtype=np.float32)

    def _precompute_showdown_data(self):
        """Cache hand scores and sorted indices for efficient P1 showdown values."""
        if len(self._board_cards) < 5:
            self._sd_scores = None
            return
        hand_ranks, _, _ = _get_tables()
        if hand_ranks is None:
            self._sd_scores = None
            return
        from submission.card_utils import combo_index_7
        board_5 = tuple(self._board_cards[-5:])
        dead = self._dead_cards

        scores = np.full(351, np.inf, dtype=np.float64)
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            if c1 not in dead and c2 not in dead:
                key = tuple(sorted((c1, c2) + board_5))
                scores[h_idx] = float(hand_ranks[combo_index_7(key)])
        # Fix P1 range hands (dead_cards includes our kept cards)
        r1_nz = np.nonzero(self._r1)[0]
        for k in r1_nz:
            if np.isinf(scores[k]):
                ck1, ck2 = ALL_HANDS[k]
                key = tuple(sorted((ck1, ck2) + board_5))
                scores[k] = float(hand_ranks[combo_index_7(key)])

        alive = ~np.isinf(scores)
        alive_idx = np.where(alive)[0]
        if len(alive_idx) < 2:
            self._sd_scores = None
            return

        # Sort alive hands by score (lower = stronger)
        alive_scores = scores[alive_idx]
        sort_order = np.argsort(alive_scores)
        sorted_alive_idx = alive_idx[sort_order]
        sorted_scores = alive_scores[sort_order]

        # Group by unique scores for tie handling
        unique_scores, group_starts = np.unique(sorted_scores, return_index=True)
        n_groups = len(unique_scores)
        group_ends = np.empty(n_groups, dtype=np.int64)
        group_ends[:-1] = group_starts[1:]
        group_ends[-1] = len(sorted_alive_idx)

        # Map each alive hand to its group index
        hand_to_group = np.zeros(len(sorted_alive_idx), dtype=np.int64)
        for g in range(n_groups):
            hand_to_group[group_starts[g]:group_ends[g]] = g

        self._sd_scores = scores
        self._sd_alive = alive
        self._sd_alive_idx = alive_idx
        self._sd_sorted_alive_idx = sorted_alive_idx
        self._sd_sort_order = sort_order
        self._sd_group_starts = group_starts
        self._sd_group_ends = group_ends
        self._sd_n_groups = n_groups
        self._sd_hand_to_group = hand_to_group

    def _compute_p1_showdown_values(self, reach_p2_terminal, chips):
        """Compute P1 per-hand showdown values using sorted prefix sums.

        Returns (351,) float32 array of P1 counterfactual values.
        reach_p2_terminal: (351,) opponent reach at this terminal.
        chips: pot / 2 (what's at stake).
        """
        out = np.zeros(351, dtype=np.float32)
        if self._sd_scores is None:
            return out

        sorted_alive_idx = self._sd_sorted_alive_idx
        sort_order = self._sd_sort_order
        alive_idx = self._sd_alive_idx
        group_starts = self._sd_group_starts
        group_ends = self._sd_group_ends
        n_groups = self._sd_n_groups
        hand_to_group = self._sd_hand_to_group

        # Get reach for sorted alive hands
        sorted_reach = reach_p2_terminal[sorted_alive_idx]

        # Compute group masses (sum of reach within each score group)
        group_masses = np.zeros(n_groups, dtype=np.float64)
        for g in range(n_groups):
            group_masses[g] = sorted_reach[group_starts[g]:group_ends[g]].sum()

        # Prefix sum of group masses
        group_prefix = np.cumsum(group_masses)
        total_reach = group_prefix[-1] if n_groups > 0 else 0.0

        if total_reach < 1e-30:
            return out

        # Tournament utility for win/lose
        tutil_win = tournament_utility(chips, self._match_margin, self._hands_remaining)
        tutil_lose = tournament_utility(-chips, self._match_margin, self._hands_remaining)

        # For each alive hand in sorted order:
        # group g: lose_mass = group_prefix[g-1] (all strictly stronger groups)
        #          win_mass = total - group_prefix[g] (all strictly weaker groups)
        for pos_in_sorted, orig_alive_pos in enumerate(sort_order):
            h = alive_idx[orig_alive_pos]
            g = hand_to_group[pos_in_sorted]

            lose_mass = group_prefix[g - 1] if g > 0 else 0.0
            win_mass = total_reach - group_prefix[g]

            out[h] = float(win_mass * tutil_win + lose_mass * tutil_lose)

        return out

    def _query_net_with_card_enumeration(self, node: SubgameNode, net) -> tuple:
        """
        Enumerate possible next cards, batch-query network, return weighted average CFVs.
        Also populates self._per_card_v2_cache.
        """
        possible_cards = [c for c in range(NUM_CARDS) if c not in self._known_cards]
        pot_frac = node.pot / max(1.0, node.pot + 2.0 * self._current_stack)
        margin_norm = self._match_margin / 500.0
        hands_norm = self._hands_remaining / 1000.0

        valid_cards, batch_p1, batch_p2 = [], [], []
        for card in possible_cards:
            p1_f = self._r1.copy()
            p2_f = self._current_opp_range.copy()
            p1_f[HAND_CONTAINS_CARD[card]] = 0.0
            p2_f[HAND_CONTAINS_CARD[card]] = 0.0
            s1, s2 = p1_f.sum(), p2_f.sum()
            if s1 == 0 or s2 == 0:
                continue
            p1_f /= s1
            p2_f /= s2
            valid_cards.append(card)
            batch_p1.append(p1_f)
            batch_p2.append(p2_f)

        if not valid_cards:
            return self._board_aware_proxy_cfv(node)

        # Check if net expects board features (river net: input_size=732)
        needs_board = hasattr(net, '_input_size') and net._input_size > 705

        batch_list = []
        for i in range(len(valid_cards)):
            # Permute ranges from solver (combo_index) to net (lex) ordering
            # combo_arr[_LEX_TO_COMBO] gathers: result[lex_idx] = combo_arr[LEX_TO_COMBO[lex_idx]]
            parts = [batch_p1[i][_LEX_TO_COMBO], batch_p2[i][_LEX_TO_COMBO],
                     np.array([pot_frac, margin_norm, hands_norm], dtype=np.float32)]
            if needs_board:
                board_oh = np.zeros(27, dtype=np.float32)
                for c in list(self._board_cards) + [valid_cards[i]]:
                    board_oh[c] = 1.0
                parts.append(board_oh)
            batch_list.append(np.concatenate(parts))
        batch_inputs = np.stack(batch_list).astype(np.float32)

        all_p1_cfv_lex, all_p2_cfv_lex = net.forward_batch(batch_inputs)

        total_p1 = np.zeros(351, dtype=np.float32)
        total_p2 = np.zeros(351, dtype=np.float32)
        for i, card in enumerate(valid_cards):
            # Permute CFVs from net (lex) back to solver (combo_index) ordering
            # lex_arr[_COMBO_TO_LEX] gathers: result[combo_idx] = lex_arr[COMBO_TO_LEX[combo_idx]]
            p2_combo = all_p2_cfv_lex[i][_COMBO_TO_LEX]
            self._per_card_v2_cache[card] = p2_combo.copy()
            total_p1 += all_p1_cfv_lex[i][_COMBO_TO_LEX]
            total_p2 += p2_combo

        n = len(valid_cards)
        return total_p1 / n, total_p2 / n


    def _hs2_proxy_cfv(self, node: SubgameNode) -> tuple:
        """Last-resort fallback: HS2-based CFV proxy (no board awareness).

        V10 FIX: Range-weighted P1 value. Instead of comparing our hand vs
        uniform opponent (hs2.mean()), weight opponent hands by actual range.
        DeepStack: "inputs to this function are the ranges for both players."
        """
        _, hs2, _ = _get_tables()
        pot = node.pot
        chips = pot / 2
        sr = node.streets_remaining
        our_h1 = self._our_hand_idx

        if sr > 0:
            implied_mult = 1.0 + sr * 0.18
        else:
            implied_mult = 1.0

        # Range-weighted opponent average hand strength
        opp_range = self._current_opp_range
        opp_range_sum = opp_range.sum()
        if opp_range_sum > 0:
            weighted_opp_hs2 = float(np.dot(opp_range, hs2) / opp_range_sum)
        else:
            weighted_opp_hs2 = float(hs2.mean())

        # P1: per-hand values for ALL alive hands (needed for per-hand P1 strategies)
        p1_chips = ((hs2 - weighted_opp_hs2) * chips * implied_mult).astype(np.float32)
        p1 = tournament_utility_vec(
            p1_chips, self._match_margin, self._hands_remaining)
        if our_h1 >= 0:
            our_hs2 = float(hs2[our_h1])
            raw_p2 = (hs2 - our_hs2) * chips
            p2_chips = (raw_p2 * implied_mult).astype(np.float32)
            p2 = tournament_utility_vec(
                p2_chips, -self._match_margin, self._hands_remaining)
        else:
            p2 = -p1.copy()
        return p1, p2

    def _compute_board_equity(self, cards_to_come: int) -> np.ndarray:
        """Compute per-hand equity vector (351,) for current board + N cards to come.

        Enumerates all possible board completions, computes exact 7-card hand
        ranks, and returns average equity per hand vs uniform opponent.
        Also computes and caches P2-vs-us: each opponent hand's outcome
        specifically against our hand, averaged over board completions.
        Cached in self._board_equity_cache keyed by cards_to_come.
        """
        if cards_to_come in self._board_equity_cache:
            return self._board_equity_cache[cards_to_come]

        hand_ranks, _, _ = _get_tables()
        if hand_ranks is None:
            return None

        from submission.equity import _combo_index_7_batch

        board = tuple(self._board_cards)
        dead = self._dead_cards
        remaining = [c for c in range(NUM_CARDS) if c not in dead]
        our_h1 = self._our_hand_idx

        # Dead-card mask: exclude hands containing dead cards
        valid_base = np.ones(351, dtype=bool)
        for c in dead:
            valid_base &= ~HAND_CONTAINS_CARD[c]
        # BUG FIX: _dead_cards includes our kept cards (for board enumeration),
        # but our hand must be valid for equity computation at showdown leaves.
        if our_h1 >= 0:
            valid_base[our_h1] = True

        total_p1_eq = np.zeros(351, dtype=np.float64)
        total_p2_vs_us = np.zeros(351, dtype=np.float64)
        n_completions = 0
        n_p2_completions = 0

        for ext in combinations(remaining, cards_to_come):
            full_board = board + ext

            # Valid hands: not colliding with known cards or extension cards
            valid_mask = valid_base.copy()
            for c in ext:
                valid_mask &= ~HAND_CONTAINS_CARD[c]
            valid_indices = np.where(valid_mask)[0]
            n_valid = len(valid_indices)
            if n_valid < 2:
                continue

            # Batch hand rank lookup
            hand_arr = np.array([ALL_HANDS[i] for i in valid_indices], dtype=np.int32)
            board_rep = np.tile(np.array(full_board, dtype=np.int32), (n_valid, 1))
            all_7 = np.concatenate([hand_arr, board_rep], axis=1)
            all_7.sort(axis=1)
            scores_valid = hand_ranks[_combo_index_7_batch(all_7)].astype(np.float64)

            # Sort-based equity: O(N log N) instead of O(N²) comparison matrix
            order = np.argsort(scores_valid)  # ascending: best hand first (lower score = better)
            sorted_scores = scores_valid[order]

            # Count wins/losses using sorted ranks, handling ties
            eq_valid = np.empty(n_valid, dtype=np.float64)
            i = 0
            while i < n_valid:
                j = i + 1
                while j < n_valid and sorted_scores[j] == sorted_scores[i]:
                    j += 1
                below = i
                above = n_valid - j
                e = (above - below) / (n_valid - 1) if n_valid > 1 else 0.0
                for k in range(i, j):
                    eq_valid[k] = e
                i = j

            eq_unsorted = np.empty(n_valid, dtype=np.float64)
            eq_unsorted[order] = eq_valid

            total_p1_eq[valid_indices] += eq_unsorted
            n_completions += 1

            # P2-vs-us: each opponent hand's outcome vs our specific hand
            if our_h1 >= 0:
                our_pos = np.where(valid_indices == our_h1)[0]
                if len(our_pos) > 0:
                    our_score = scores_valid[our_pos[0]]
                    # P2 wins if their score < ours (stronger), loses if > ours
                    p2_outcome = np.where(
                        scores_valid < our_score, 1.0,
                        np.where(scores_valid > our_score, -1.0, 0.0))
                    total_p2_vs_us[valid_indices] += p2_outcome
                    n_p2_completions += 1

        if n_completions == 0:
            self._board_equity_cache[cards_to_come] = None
            return None

        equity = (total_p1_eq / n_completions).astype(np.float32)
        self._board_equity_cache[cards_to_come] = equity

        # Cache P2-vs-us outcome
        if n_p2_completions > 0:
            self._board_equity_cache[('p2_vs_us', cards_to_come)] = \
                (total_p2_vs_us / n_p2_completions).astype(np.float32)

        return equity

    def _board_aware_proxy_cfv(self, node: SubgameNode) -> tuple:
        """Board-aware CFV using cached equity vector × pot scaling.

        V10 FIX: Range-weighted P1 value. P1's equity is now computed as
        the weighted average outcome vs each opponent hand, weighted by
        the actual opponent range (not uniform). This matches DeepStack's
        approach where leaf values depend on both players' ranges.

        P2 CFV uses per-hand outcome vs our specific hand (unchanged).
        """
        equity = self._compute_board_equity(node.streets_remaining)
        if equity is None:
            return self._hs2_proxy_cfv(node)

        chips = node.pot / 2

        # P2: use per-hand outcome vs our specific hand
        p2_vs_us = self._board_equity_cache.get(
            ('p2_vs_us', node.streets_remaining))
        if p2_vs_us is not None:
            sr = node.streets_remaining
            if sr > 0:
                implied_mult = 1.0 + sr * 0.18
                p2_implied = (p2_vs_us * chips * implied_mult).astype(np.float32)
            else:
                p2_implied = (p2_vs_us * chips).astype(np.float32)

            p2_cfv = tournament_utility_vec(
                p2_implied, -self._match_margin, self._hands_remaining)
        else:
            return self._hs2_proxy_cfv(node)

        # P1: per-hand values for ALL alive hands (needed for per-hand P1 strategies)
        opp_range = self._current_opp_range
        opp_range_sum = opp_range.sum()

        sr = node.streets_remaining
        implied_mult = 1.0 + sr * 0.18 if sr > 0 else 1.0

        if opp_range_sum > 0:
            weighted_opp_eq = float(np.dot(opp_range, equity) / opp_range_sum)
        else:
            weighted_opp_eq = float(equity.mean())
        p1_chips = ((equity - weighted_opp_eq) * chips * implied_mult).astype(np.float32)
        p1_cfv = tournament_utility_vec(
            p1_chips, self._match_margin, self._hands_remaining)
        return p1_cfv, p2_cfv

    def _exact_river_cfv(self, node: SubgameNode) -> tuple:
        """
        Exact showdown CFV at river using HAND_RANKS table (7-card lookup).
        P2 CFV = weighted average matchup over P1's widened range.
        P1 CFV = cached for our actual hand (diagnostics).
        Returns (p1_cfv, p2_cfv) each shape (351,).
        """
        hand_ranks, _, _ = _get_tables()
        if hand_ranks is None or len(self._board_cards) < 5:
            return self._hs2_proxy_cfv(node)

        our_h1 = self._our_hand_idx
        if our_h1 < 0:
            return self._hs2_proxy_cfv(node)

        from submission.card_utils import combo_index_7
        board_5 = tuple(self._board_cards[-5:])
        dead = self._dead_cards

        scores = np.full(351, np.inf)
        for h_idx, (c1, c2) in enumerate(ALL_HANDS):
            if c1 not in dead and c2 not in dead:
                key = tuple(sorted((c1, c2) + board_5))
                scores[h_idx] = float(hand_ranks[combo_index_7(key)])

        # BUG FIX: _dead_cards includes our kept cards (for board enumeration),
        # but P1 range hands must have valid showdown scores.
        r1 = self._r1
        r1_nz = np.nonzero(r1)[0]
        for k in r1_nz:
            if np.isinf(scores[k]):
                ck1, ck2 = ALL_HANDS[k]
                key = tuple(sorted((ck1, ck2) + board_5))
                scores[k] = float(hand_ranks[combo_index_7(key)])

        alive = ~np.isinf(scores)
        alive_mask = alive  # bool array
        chips = node.pot / 2

        # P2 CFV: weighted average matchup over P1's widened range
        # For each P1 hand k: P2 wins if P2 score < k_score (lower = stronger)
        p2_outcome = np.zeros(351, dtype=np.float32)
        scores_alive = scores[alive_mask]
        for k in r1_nz:
            k_score = scores[k]
            if np.isinf(k_score):
                continue
            k_out = np.where(
                scores_alive < k_score, 1.0,
                np.where(scores_alive > k_score, -1.0, 0.0)
            ).astype(np.float32)
            temp = np.zeros(351, dtype=np.float32)
            temp[alive_mask] = k_out
            p2_outcome += r1[k] * temp

        # Tournament Utility (MWP) instead of raw chips
        p2_chips_won = p2_outcome * chips
        p2_cfv = tournament_utility_vec(
            p2_chips_won, -self._match_margin, self._hands_remaining)

        # P1 CFV for ALL alive hands (needed for per-hand P1 strategies)
        n_alive = int(alive.sum())
        p1_cfv = np.zeros(351, dtype=np.float32)
        if n_alive > 1:
            for h_idx in r1_nz:
                h_score = scores[h_idx]
                if np.isinf(h_score):
                    continue
                h_wins = float(np.sum(scores[alive] > h_score))
                h_losses = float(np.sum(scores[alive] < h_score))
                h_eq = (h_wins - h_losses) / (n_alive - 1)
                p1_cfv[h_idx] = tournament_utility(
                    h_eq * chips, self._match_margin, self._hands_remaining)
        return p1_cfv, p2_cfv

    def _cfrd_root_strategy(self) -> np.ndarray:
        """Return opponent range for P1 terminal value computation.
        Uses Bayesian reconstructed range as prior for the CFR-D gadget.
        """
        return self._current_opp_range.copy()

    def _iterate_cfrd(self) -> None:
        """Run DCFR iterations until budget exhausted (flat iterative).

        V10 FIX: Replaced CFR+ with DCFR (alpha=1.5, beta=0, gamma=2).
        Brown & Sandholm 2019: "setting alpha=3/2, beta=0, and gamma=2 led to
        performance that was consistently stronger than CFR+."

        DCFR discounts prior iterations so early random exploration doesn't
        permanently pollute the average strategy. With CFR+, if FOLD gets
        explored randomly on iteration 1, it takes ~471k iterations to wash
        out. DCFR achieves this in ~970. Critical for our time-limited solver.

        Discount formulas per iteration t:
          positive regrets *= t^alpha / (t^alpha + 1)  [alpha=1.5]
          negative regrets *= t^beta / (t^beta + 1)    [beta=0 → always 0.5]
          strategy sum weight = (t/(t+1))^gamma         [gamma=2]

        Per-hand P1 strategies: each phantom hand develops its own strategy,
        so nut hands raise and air hands fold. P2 faces realistic action
        distributions rather than uniform phantom play.
        """
        # DCFR parameters (Brown & Sandholm recommended)
        DCFR_ALPHA = 1.5
        DCFR_GAMMA = 2.0
        # beta=0 means discount factor = t^0/(t^0+1) = 0.5 always
        DCFR_NEG_DISCOUNT = 0.5

        # Local references for speed
        N = self._flat_N
        root_idx = self._flat_root_idx
        is_terminal = self._flat_is_terminal
        is_fold_terminal = self._flat_is_fold_terminal
        player_arr = self._flat_player
        actions_arr = self._flat_actions
        child_idx_arr = self._flat_child_idx
        n_children_arr = self._flat_n_children
        regret_sum_arr = self._flat_regret_sum
        strategy_sum_arr = self._flat_strategy_sum
        p2 = self._flat_p2_all    # (N, 351) pre-allocated
        p1 = self._flat_p1_all    # (N, 351) per-hand P1 values
        reach_p1 = self._flat_reach_p1  # (N, 351) per-hand P1 reach
        reach_p2 = self._flat_reach_p2  # (N, 351) pre-allocated
        sv_buf = self._flat_sv
        last_cfvs = self._last_cfvs
        dot = np.dot
        _max = max  # local ref to builtin
        is_p2_decision = self._flat_is_p2_decision
        is_p1_decision = self._flat_is_p1_decision
        p2_regret_arr = self._flat_p2_regret
        p2_strat_ph_arr = self._flat_p2_strat_ph
        p2_strat_sum_arr = self._flat_p2_strat_sum
        p2_model_sv_arr = self._flat_p2_model_sv
        p1_regret_arr = self._flat_p1_regret
        p1_strat_ph_arr = self._flat_p1_strat_ph
        p1_strat_sum_arr = self._flat_p1_strat_sum
        p2_scratch = self._flat_p2_scratch  # (351,) scratch buffer for RNR
        rnr_p = self._rnr_p
        one_minus_p = 1.0 - rnr_p
        nodes = self._flat_nodes
        our_h = self._our_hand_idx

        # Precompute showdown data for P1 per-hand terminal values
        self._precompute_showdown_data()
        has_showdown = self._sd_scores is not None

        # Precompute P1 fold terminal values (constant chip payoff for all hands)
        p1_fold_vals = [None] * N
        for i in range(N):
            if is_terminal[i] and is_fold_terminal[i]:
                nd = nodes[i]
                if hasattr(nd, '_cached_p1_cfv') and nd._cached_p1_cfv is not None:
                    p1_fold_vals[i] = float(nd._cached_p1_cfv[0])

        # Precompute P1 street-boundary terminal values
        p1_boundary_vals = [None] * N
        for i in range(N):
            if is_terminal[i] and not is_fold_terminal[i]:
                nd = nodes[i]
                if hasattr(nd, '_cached_p1_cfv') and nd._cached_p1_cfv is not None:
                    if nd.streets_remaining > 0:
                        p1_boundary_vals[i] = nd._cached_p1_cfv

        while time.time() < self._budget_end:
            for _ in range(20):
                opp_range = self._cfrd_root_strategy()
                t = self._iterations_run + 1

                # DCFR discount factors for this iteration
                t_alpha = t ** DCFR_ALPHA
                pos_discount = t_alpha / (t_alpha + 1.0)  # discount for positive regrets
                neg_discount = DCFR_NEG_DISCOUNT           # 0.5 for beta=0
                strat_weight = (t / (t + 1.0)) ** DCFR_GAMMA  # strategy sum weight

                # Set root reaches
                reach_p1[root_idx] = self._r1  # P1 range (per-hand reach)
                reach_p2[root_idx] = 1.0  # Broadcasts to (351,)

                # === Top-down pass (reverse post-order = pre-order) ===
                for i in range(N - 1, -1, -1):
                    if is_terminal[i]:
                        continue
                    actions = actions_arr[i]
                    child_indices = child_idx_arr[i]
                    n_ch = n_children_arr[i]
                    rp1 = reach_p1[i]  # (351,) per-hand
                    rp2 = reach_p2[i]

                    if is_p2_decision[i]:
                        # P2 node: per-hand strategy from per-hand regrets
                        p2_rs_i = p2_regret_arr[i]  # (351, N_ACTIONS)
                        pos = np.maximum(p2_rs_i[:, actions], 0)  # (351, n_ch)
                        totals = pos.sum(axis=1, keepdims=True)   # (351, 1)
                        u = 1.0 / n_ch
                        strat_ph = np.where(totals > 0, pos / np.maximum(totals, 1e-30), u)
                        # Store CFR strategy into full (351, N_ACTIONS) for bottom-up
                        p2_sp_i = p2_strat_ph_arr[i]
                        p2_sp_i[:] = 0
                        for j in range(n_ch):
                            p2_sp_i[:, actions[j]] = strat_ph[:, j]

                        # RNR: blend with model strategy for reach propagation
                        model_sv_i = p2_model_sv_arr[i]
                        sv_buf[i] = None
                        if rnr_p > 0.01 and model_sv_i is not None:
                            for j in range(n_ch):
                                ci = child_indices[j]
                                reach_p1[ci] = rp1  # (351,) copied
                                # eff = (1-p)*cfr + p*model
                                reach_p2[ci] = rp2 * (one_minus_p * strat_ph[:, j] + rnr_p * model_sv_i[j])
                        else:
                            for j in range(n_ch):
                                ci = child_indices[j]
                                reach_p1[ci] = rp1
                                reach_p2[ci] = rp2 * strat_ph[:, j]
                    else:
                        # P1 node: per-hand strategy (regret matching per hand)
                        p1_rs = p1_regret_arr[i]  # (351, N_ACTIONS)
                        p1_sp = p1_strat_ph_arr[i]  # (351, N_ACTIONS) scratch
                        pos = np.maximum(p1_rs[:, actions], 0)  # (351, n_ch)
                        totals = pos.sum(axis=1, keepdims=True)  # (351, 1)
                        u = 1.0 / n_ch
                        strat = np.where(totals > 0, pos / np.maximum(totals, 1e-30), u)
                        p1_sp[:] = 0
                        for j in range(n_ch):
                            p1_sp[:, actions[j]] = strat[:, j]

                        # Range-weighted P1 strategy for P2's value propagation
                        # P2 doesn't know P1's hand, so sees the range-weighted average
                        range_total = rp1.sum()
                        if range_total > 1e-30:
                            range_strat = (rp1[:, np.newaxis] * strat).sum(axis=0) / range_total  # (n_ch,)
                        else:
                            range_strat = np.full(n_ch, 1.0 / n_ch)
                        sv_buf[i] = [float(range_strat[j]) for j in range(n_ch)]

                        # Per-hand P1 reach propagation
                        for j in range(n_ch):
                            ci = child_indices[j]
                            reach_p1[ci] = rp1 * strat[:, j]  # (351,) per-hand
                            reach_p2[ci] = rp2

                # === Compute P1 terminal values ===
                # P1 CFV(h1) = Σ_{h2} (opp_range * reach_p2)[h2] * payoff(h1, h2)
                for i in range(N):
                    if not is_terminal[i]:
                        continue
                    narrowed_p2 = opp_range * reach_p2[i]  # (351,)
                    if p1_fold_vals[i] is not None:
                        # Fold terminal: constant value * total opponent reach
                        p1[i][:] = p1_fold_vals[i] * narrowed_p2.sum()
                    elif p1_boundary_vals[i] is not None:
                        # Street boundary: value-net estimated
                        p1[i][:] = p1_boundary_vals[i] * narrowed_p2.sum()
                    elif has_showdown and not is_fold_terminal[i]:
                        nd = nodes[i]
                        if nd.streets_remaining == 0:
                            chips = nd.pot / 2
                            p1[i] = self._compute_p1_showdown_values(narrowed_p2, chips)
                    else:
                        p1[i][:] = 0.0

                # === Bottom-up pass (post-order) ===
                for i in range(N):
                    if is_terminal[i]:
                        continue

                    actions = actions_arr[i]
                    child_indices = child_idx_arr[i]
                    n_ch = n_children_arr[i]

                    if is_p2_decision[i]:
                        # P2 node: per-hand values, regrets, strategy sums
                        p2_sp_i = p2_strat_ph_arr[i]  # (351, N_ACTIONS) — CFR strategy
                        model_sv_i = p2_model_sv_arr[i]
                        a0 = actions[0]
                        rnr_active = rnr_p > 0.01 and model_sv_i is not None

                        if rnr_active:
                            # RNR active: compute p2_free into scratch buffer
                            p2_free = p2_scratch
                            np.multiply(p2_sp_i[:, a0], p2[child_indices[0]], out=p2_free)
                            for j in range(1, n_ch):
                                p2_free += p2_sp_i[:, actions[j]] * p2[child_indices[j]]

                            # Locked-branch value
                            p2_locked = model_sv_i[0] * p2[child_indices[0]]
                            for j in range(1, n_ch):
                                p2_locked = p2_locked + model_sv_i[j] * p2[child_indices[j]]

                            p2[i][:] = one_minus_p * p2_free + rnr_p * p2_locked
                        else:
                            p2_free = p2[i]
                            np.multiply(p2_sp_i[:, a0], p2[child_indices[0]], out=p2_free)
                            for j in range(1, n_ch):
                                p2_free += p2_sp_i[:, actions[j]] * p2[child_indices[j]]

                        # P2 per-hand regret update (DCFR)
                        rp1_sum = reach_p1[i].sum()  # scalar: total P1 reach
                        regret_scale = one_minus_p * rp1_sum if rnr_active else rp1_sum
                        p2_rs_i = p2_regret_arr[i]
                        for j in range(n_ch):
                            a = actions[j]
                            delta = p2[child_indices[j]] - p2_free
                            existing = p2_rs_i[:, a]
                            discounted = np.where(
                                existing > 0,
                                existing * pos_discount,
                                existing * neg_discount
                            )
                            p2_rs_i[:, a] = discounted + regret_scale * delta

                        # Per-hand strategy sum
                        p2_ss_i = p2_strat_sum_arr[i]
                        for j in range(n_ch):
                            p2_ss_i[:, actions[j]] += strat_weight * p2_sp_i[:, actions[j]]

                        # P1 value at P2 node: sum of children (reach_p2 carries weights)
                        p1[i][:] = p1[child_indices[0]]
                        for j in range(1, n_ch):
                            p1[i] += p1[child_indices[j]]

                    else:
                        # P1 node: per-hand strategy
                        p1_sp = p1_strat_ph_arr[i]  # (351, N_ACTIONS) from top-down
                        sv = sv_buf[i]  # range-weighted strategy for P2

                        # P2 value: range-weighted P1 strategy
                        p2_i = p2[i]
                        np.multiply(sv[0], p2[child_indices[0]], out=p2_i)
                        for j in range(1, n_ch):
                            p2_i += sv[j] * p2[child_indices[j]]

                        # P1 per-hand value: weighted sum of child values by per-hand strategy
                        a0 = actions[0]
                        np.multiply(p1_sp[:, a0], p1[child_indices[0]], out=p1[i])
                        for j in range(1, n_ch):
                            p1[i] += p1_sp[:, actions[j]] * p1[child_indices[j]]

                        # Root CFVs: use per-hand P1 values for our hand
                        if i == root_idx:
                            last_cfvs[:] = 0.0
                            if our_h >= 0:
                                for j in range(n_ch):
                                    last_cfvs[actions[j]] = float(p1[child_indices[j]][our_h])
                            else:
                                # Fallback: range-averaged
                                node_opp_range = opp_range * reach_p2[i]
                                for j in range(n_ch):
                                    last_cfvs[actions[j]] = -float(dot(node_opp_range, p2[child_indices[j]]))

                        # P1 per-hand regret update (DCFR)
                        p1_rs = p1_regret_arr[i]
                        for j in range(n_ch):
                            a = actions[j]
                            delta = p1[child_indices[j]] - p1[i]  # (351,) per-hand regret
                            existing = p1_rs[:, a]
                            discounted = np.where(
                                existing > 0,
                                existing * pos_discount,
                                existing * neg_discount
                            )
                            p1_rs[:, a] = discounted + delta

                        # P1 per-hand strategy sum
                        p1_ss = p1_strat_sum_arr[i]
                        for j in range(n_ch):
                            p1_ss[:, actions[j]] += strat_weight * p1_sp[:, actions[j]]

                        # Scalar regret/strategy sum (backward compat with avg_strategy)
                        node_opp_range = opp_range * reach_p2[i]
                        node_p1_val = -float(dot(node_opp_range, p2_i))
                        rs = regret_sum_arr[i]
                        for j in range(n_ch):
                            action_p1_val = -float(dot(node_opp_range, p2[child_indices[j]]))
                            regret = action_p1_val - node_p1_val
                            a = actions[j]
                            existing = float(rs[a])
                            if existing > 0:
                                rs[a] = existing * pos_discount + regret
                            else:
                                rs[a] = existing * neg_discount + regret

                        ss = strategy_sum_arr[i]
                        for j in range(n_ch):
                            ss[actions[j]] += strat_weight * sv[j]

                self._last_opp_subgame_cfv = p2[root_idx].copy()
                self._iterations_run += 1


    def _extract_hand_specific_cfvs(self) -> None:
        """Pass 2: Compute hand-specific CFVs using P2's converged strategy.

        After CFR with widened P1 range (pass 1), P2's strategy has converged
        against the full range. Pass 2 re-evaluates P1's options by propagating
        P2 values bottom-up using the converged AVERAGE strategy (more stable
        than the last iteration's noisy strategy). The root CFVs then reflect
        P1's expected value with the converged equilibrium opponent.
        """
        if not hasattr(self, '_flat_N') or self._flat_N == 0:
            return
        our_h1 = self._our_hand_idx
        if our_h1 < 0:
            return

        N = self._flat_N
        root_idx = self._flat_root_idx
        is_terminal = self._flat_is_terminal
        is_p2_decision = self._flat_is_p2_decision
        actions_arr = self._flat_actions
        child_idx_arr = self._flat_child_idx
        n_children_arr = self._flat_n_children
        p2_strat_sum_arr = self._flat_p2_strat_sum
        strategy_sum_arr = self._flat_strategy_sum

        # Allocate pass-2 buffer
        p2_pm = np.zeros((N, 351), dtype=np.float32)

        # Terminal values: start from pass-1 values, then correct ALL terminals
        is_fold_terminal = self._flat_is_fold_terminal
        for i in range(N):
            if is_terminal[i]:
                p2_pm[i] = self._flat_p2_all[i]

        # Non-river depth-limit correction: pass-1 terminal values are
        # range-averaged (value net queried with full P1 range). Replace
        # with hand-specific equity: each opponent hand's outcome vs our
        # actual hand, averaged over remaining board cards.
        max_cards_to_come = 5 - len(self._board_cards)
        for sr in range(1, max_cards_to_come + 1):
            p2_vs_us = self._board_equity_cache.get(('p2_vs_us', sr))
            if p2_vs_us is None:
                try:
                    self._compute_board_equity(sr)
                    p2_vs_us = self._board_equity_cache.get(('p2_vs_us', sr))
                except Exception:
                    continue
            if p2_vs_us is None:
                continue
            for i in range(N):
                if not is_terminal[i] or is_fold_terminal[i]:
                    continue
                nd = self._flat_nodes[i]
                if nd.streets_remaining != sr:
                    continue
                chips = nd.pot / 2
                p2_pm[i] = tournament_utility_vec(
                    p2_vs_us * chips, -self._match_margin,
                    self._hands_remaining)

        # River showdown correction: recompute P2 payoffs vs our actual hand
        # (pass-1 values are range-weighted, diluted by phantom hands)
        if len(self._board_cards) >= 5:
            try:
                hand_ranks, _, _ = _get_tables()
                if hand_ranks is not None:
                    from submission.card_utils import combo_index_7
                    board_5 = tuple(self._board_cards[-5:])
                    dead = self._dead_cards
                    scores = np.full(351, np.inf)
                    for h_idx, (c1, c2) in enumerate(ALL_HANDS):
                        if c1 not in dead and c2 not in dead:
                            key = tuple(sorted((c1, c2) + board_5))
                            scores[h_idx] = float(hand_ranks[combo_index_7(key)])
                    # Fix our hand's score (dead_cards includes our kept cards)
                    if np.isinf(scores[our_h1]):
                        ck1, ck2 = ALL_HANDS[our_h1]
                        key = tuple(sorted((ck1, ck2) + board_5))
                        scores[our_h1] = float(hand_ranks[combo_index_7(key)])
                    our_score = scores[our_h1]
                    alive = ~np.isinf(scores)

                    for i in range(N):
                        if not is_terminal[i] or is_fold_terminal[i]:
                            continue
                        nd = self._flat_nodes[i]
                        if nd.streets_remaining != 0:
                            continue
                        # Showdown: P2 payoff vs our actual hand (point-mass)
                        chips = nd.pot / 2
                        p2_out = np.zeros(351, dtype=np.float32)
                        p2_out[alive] = np.where(
                            scores[alive] < our_score, 1.0,
                            np.where(scores[alive] > our_score, -1.0, 0.0)
                        ).astype(np.float32)
                        p2_pm[i] = tournament_utility_vec(
                            p2_out * chips, -self._match_margin,
                            self._hands_remaining)
            except Exception:
                pass  # fall through to pass-1 values

        # Opponent range for P1 best-response computation
        opp_range = self._current_opp_range.copy()

        # --- Top-down pass: propagate opponent ranges through tree ---
        # At P2 decision nodes, narrow the opponent range by P2's converged
        # strategy for each action. This accounts for the fact that opponents
        # who bet have different hand distributions than those who check.
        node_opp_range = np.zeros((N, 351), dtype=np.float32)
        node_opp_range[root_idx] = opp_range

        # Reverse of post-order = top-down (parents before children).
        # Start from root_idx (not N-1) so that when observe_action changes
        # root_idx to a subtree root, we don't process nodes above it that
        # would overwrite our opp_range with zeros.
        for i in range(root_idx, -1, -1):
            if is_terminal[i]:
                continue
            actions = actions_arr[i]
            child_indices = child_idx_arr[i]
            n_ch = n_children_arr[i]

            if is_p2_decision[i]:
                # P2 node: narrow range by P2's average strategy per action
                p2_ss = p2_strat_sum_arr[i]  # (351, N_ACTIONS)
                action_list = list(actions)
                totals = p2_ss[:, action_list].sum(axis=1, keepdims=True)
                uniform_p = 1.0 / n_ch
                strat = np.where(totals > 0,
                                 p2_ss[:, action_list] / np.maximum(totals, 1e-30),
                                 uniform_p)
                for j in range(n_ch):
                    child_range = node_opp_range[i] * strat[:, j]
                    s = child_range.sum()
                    if s > 0:
                        child_range /= s
                    else:
                        child_range = node_opp_range[i].copy()
                    node_opp_range[child_indices[j]] = child_range
            else:
                # P1 node: our action doesn't reveal opponent's hand
                for j in range(n_ch):
                    node_opp_range[child_indices[j]] = node_opp_range[i]

        # --- Bottom-up propagation with CONVERGED average strategies ---
        # Only process nodes in the subtree (0..root_idx) for efficiency.
        for i in range(root_idx + 1):
            if is_terminal[i]:
                continue
            actions = actions_arr[i]
            child_indices = child_idx_arr[i]
            n_ch = n_children_arr[i]

            if is_p2_decision[i]:
                # P2: use converged average strategy
                p2_ss = p2_strat_sum_arr[i]  # (351, N_ACTIONS)
                action_list = list(actions)
                totals = p2_ss[:, action_list].sum(axis=1, keepdims=True)
                uniform_p = 1.0 / n_ch
                strat = np.where(totals > 0,
                                 p2_ss[:, action_list] / np.maximum(totals, 1e-30),
                                 uniform_p)
                p2_pm[i] = strat[:, 0] * p2_pm[child_indices[0]]
                for j in range(1, n_ch):
                    p2_pm[i] += strat[:, j] * p2_pm[child_indices[j]]
            else:
                # P1: best response using narrowed opponent range at this node.
                # After P2 actions above, node_opp_range[i] reflects which
                # opponent hands actually reach this point in the tree.
                best_val = -1e30
                best_j = 0
                nr = node_opp_range[i]
                for j in range(n_ch):
                    val_j = -float(np.dot(nr, p2_pm[child_indices[j]]))
                    if val_j > best_val:
                        best_val = val_j
                        best_j = j
                p2_pm[i] = p2_pm[child_indices[best_j]].copy()

        # Extract hand-specific CFVs at root
        root_actions = actions_arr[root_idx]
        root_children = child_idx_arr[root_idx]
        self._hand_cfvs[:] = 0.0
        root_nr = node_opp_range[root_idx]
        if root_actions is not None:
            for j in range(len(root_actions)):
                self._hand_cfvs[root_actions[j]] = -float(
                    np.dot(root_nr, p2_pm[root_children[j]]))


    def get_action_cfvs(self) -> np.ndarray:
        """Return hand-specific action CFVs from pass 2, shape (N_ACTIONS,) float32."""
        # Prefer pass-2 hand-specific CFVs; fall back to pass-1 range-averaged
        if np.any(self._hand_cfvs != 0):
            return self._hand_cfvs.copy()
        return self._last_cfvs.copy()

    def get_root_average_strategy(self) -> np.ndarray:
        """Return average strategy at current root, shape (N_ACTIONS,) float32.
        Prefers per-hand P1 strategy for our specific hand when available."""
        if self._current_root is None:
            return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
        # Try per-hand P1 strategy for our specific hand
        our_h = self._our_hand_idx
        if our_h >= 0 and hasattr(self, '_flat_node_map') and hasattr(self, '_flat_p1_strat_sum'):
            root_flat_idx = self._flat_node_map.get(id(self._current_root))
            if root_flat_idx is not None:
                p1_ss = self._flat_p1_strat_sum[root_flat_idx]
                if p1_ss is not None:
                    valid = list(self._current_root.children.keys())
                    if valid:
                        total = p1_ss[our_h, valid].sum()
                        if total > 0:
                            s = np.zeros(N_ACTIONS, dtype=np.float32)
                            s[valid] = p1_ss[our_h, valid] / total
                            return s
        # Fallback to scalar average strategy
        return self._current_root.get_average_strategy()

    def observe_action(self, abstract_action_idx: int) -> None:
        """Advance current root to child for given action. Re-runs CFR on subtree."""
        if (self._current_root is None or
                abstract_action_idx not in self._current_root.children):
            return

        # Fix #15: Narrow opponent range when opponent acts (P2 node).
        # When the opponent takes an action, hands that would rarely choose
        # that action should get less weight. E.g., opponent shoves all-in →
        # weak hands that never shove get near-zero weight → our equity
        # against the narrowed range drops → we fold more correctly.
        old_root = self._current_root
        if (old_root.player == 1 and hasattr(self, '_flat_node_map')
                and hasattr(self, '_flat_p2_strat_sum')):
            root_flat_idx = self._flat_node_map.get(id(old_root))
            if root_flat_idx is not None:
                p2_ss = self._flat_p2_strat_sum[root_flat_idx]
                if p2_ss is not None:
                    actions = self._flat_actions[root_flat_idx]
                    if actions is not None and abstract_action_idx in actions:
                        col = list(actions).index(abstract_action_idx)
                        # Per-hand probability of choosing this action
                        hand_totals = p2_ss[:, list(actions)].sum(axis=1)
                        action_prob = np.where(
                            hand_totals > 0,
                            p2_ss[:, abstract_action_idx] / np.maximum(hand_totals, 1e-30),
                            1.0 / len(actions))  # uniform fallback
                        self._current_opp_range *= action_prob
                        s = self._current_opp_range.sum()
                        if s > 0:
                            self._current_opp_range /= s

        child = self._current_root.children[abstract_action_idx]
        self._current_pot = child.pot
        self._current_stack = child.stack
        self._current_root = child
        if self._budget_end > time.time() and child.children:
            self._prepare_flat_traversal(child)
            self._iterate_cfrd()
            self._extract_hand_specific_cfvs()  # pass 2
        elif child.children and hasattr(self, '_flat_node_map'):
            # Budget expired — extract pass-2 CFVs from existing flat arrays.
            # The P2 strategies from the parent solve are still valid; we just
            # need to re-extract at the new tree position. Cost: ~1ms.
            child_flat_idx = self._flat_node_map.get(id(child))
            if child_flat_idx is not None:
                self._flat_root_idx = child_flat_idx
                self._extract_hand_specific_cfvs()
            else:
                # Node not in flat map (shouldn't happen within a street)
                self._hand_cfvs[:] = 0.0

    def update_for_new_street(self, new_card: int,
                              real_pot: float = None,
                              real_stack: float = None,
                              hand_actions: list = None,
                              opp_profile=None) -> None:
        """
        Called by player.py when turn or river card is revealed.
        Filters r1, updates v2 from cache, rebuilds tree, re-runs CFR.
        """
        # _our_hand_idx is stable (our kept hand never contains a community card)
        # Filter r1 for new card (phantoms containing this card are removed)
        self._r1[HAND_CONTAINS_CARD[new_card]] = 0.0
        s = self._r1.sum()
        if s > 0:
            self._r1 /= s

        if new_card in self._per_card_v2_cache:
            self._v2 = self._per_card_v2_cache[new_card]

        self._known_cards.add(new_card)
        self._dead_cards.add(new_card)  # new board card is dead for equity
        self._board_cards.append(new_card)
        self._streets_remaining -= 1

        # Use real pot/stack from game engine if provided (more accurate than
        # abstract tree tracking which can diverge from real game state)
        pot = real_pot if real_pot is not None else self._current_pot
        stack = real_stack if real_stack is not None else self._current_stack
        self._current_pot = pot
        self._current_stack = stack

        # Refresh opponent tendencies for new street (continuous, no thresholds)
        if opp_profile is not None:
            try:
                opp_ctx = opp_profile.get_context(1)
                # Use postflop fold rate when available
                if opp_ctx.get('postflop_raises_faced', 0) >= 3:
                    raw_fold = float(opp_ctx.get('postflop_fold_rate', 0.30))
                else:
                    raw_fold = float(opp_ctx.get('fold_rate_to_our_raise', 0.30))
                self._opp_fold_rate = max(0.05, min(0.95, raw_fold))
                self._opp_raise_rate = max(0.05, min(0.95,
                    float(opp_ctx.get('raise_rate', 0.30))))
                self._opp_vpip = float(opp_ctx.get('vpip', 1.0))
                self._opp_confidence = min(opp_profile.hands_observed / 50.0, 1.0)
                deviation = abs(self._opp_fold_rate - 0.30)
                self._rnr_p = min(0.50, deviation * self._opp_confidence)
            except Exception:
                pass

        root_player = 1 if self._in_position else 0
        self._root = self._build_tree(
            player=root_player,
            pot=pot,
            stack=stack,
            ip_in_pot=pot / 2,
            oop_in_pot=pot / 2,
            action_history=(),
            raises_this_street=0,
            streets_remaining=self._streets_remaining,
        )
        self._current_root = self._root

        self._last_opp_subgame_cfv = self._v2.copy()
        self._per_card_v2_cache = {}
        self._board_equity_cache = {}

        # Rebuild widened P1 range with fresh phantoms for the new board
        cards_to_come = 5 - len(self._board_cards)
        _phantom_eq = self._compute_board_equity(cards_to_come)
        if _phantom_eq is not None:
            self._r1 = self._build_widened_range(_phantom_eq, self._streets_remaining)
        else:
            # Fallback: point-mass on our actual hand
            self._r1 = np.zeros(351, dtype=np.float32)
            if self._our_hand_idx >= 0:
                self._r1[self._our_hand_idx] = 1.0
        # Keep existing opponent range (post-discard weighted), just filter
        # for the new card. Don't reset to uniform — that loses all the
        # information about what hands the opponent would actually hold.
        self._current_opp_range[HAND_CONTAINS_CARD[new_card]] = 0.0
        s = self._current_opp_range.sum()
        if s > 0:
            self._current_opp_range /= s

        # Bayesian update: narrow opponent range based on their betting actions
        # from previous streets. If they raised the flop, strong hands are more
        # likely; if they checked, weak hands are more likely.
        if hand_actions and opp_profile:
            try:
                from submission.equity import make_action_weights_fn
                nonzero_idx = np.nonzero(self._current_opp_range)[0]
                if len(nonzero_idx) > 0:
                    candidate_hands = [ALL_HANDS[i] for i in nonzero_idx]
                    weights_fn = make_action_weights_fn(
                        opp_profile.opp_weights_fn, candidate_hands,
                        list(self._board_cards), hand_actions)
                    if weights_fn is not None:
                        for i in nonzero_idx:
                            self._current_opp_range[i] *= float(weights_fn(ALL_HANDS[i]))
                        s = self._current_opp_range.sum()
                        if s > 0:
                            self._current_opp_range /= s
            except Exception:
                pass

        # Continuous opponent range weighting on new street
        self._apply_continuous_range_weighting()

        self._precompute_depth_limit(self._root)
        self._prepare_flat_traversal(self._root)

        self._iterations_run = 0
        # Use street-appropriate budget
        if self._streets_remaining == 2:
            budget = self._budget_turn
        elif self._streets_remaining == 1:
            budget = self._budget_river
        else:
            budget = self._budget_flop
        self._budget_per_street = budget
        self._budget_end = time.time() + budget
        self._iterate_cfrd()
        self._extract_hand_specific_cfvs()  # pass 2: hand-specific CFVs

    def update_opponent_discards(self, opp_discards: tuple) -> None:
        """Filter current_opp_range and mark discards as dead cards."""
        for c in opp_discards:
            if 0 <= c < NUM_CARDS:
                self._current_opp_range[HAND_CONTAINS_CARD[c]] = 0.0
                self._dead_cards.add(c)
                self._known_cards.add(c)
        s = self._current_opp_range.sum()
        if s > 0:
            self._current_opp_range /= s

    def _get_raise_size_distribution(self) -> dict:
        """Compute raise action distribution from observed opponent bet sizes."""
        profile = self._opp_profile
        if profile is not None and hasattr(profile, '_raise_sizes') and len(profile._raise_sizes) >= 5:
            counts = {RAISE_SMALL: 0, RAISE_LARGE: 0, RAISE_OVERBET: 0, RAISE_ALLIN: 0}
            for bet_size, pot in profile._raise_sizes:
                frac = bet_size / max(1, pot)
                if frac <= 0.50:
                    counts[RAISE_SMALL] += 1
                elif frac <= 0.90:
                    counts[RAISE_LARGE] += 1
                elif frac <= 1.30:
                    counts[RAISE_OVERBET] += 1
                else:
                    counts[RAISE_ALLIN] += 1
            total = sum(counts.values())
            if total > 0:
                return {k: v / total for k, v in counts.items()}
        # Default: weight toward smaller raises (most heuristic bots)
        return {RAISE_SMALL: 0.45, RAISE_LARGE: 0.30, RAISE_OVERBET: 0.15, RAISE_ALLIN: 0.10}

    def _compute_opp_model_sv(self, actions: tuple, facing_raise: bool) -> list:
        """Compute opponent model strategy from observed action frequencies.

        Uses per-action postflop counts from OpponentModel when sufficient
        data exists (>=5 observations), falling back to aggregate rates.
        """
        raise_dist = self._get_raise_size_distribution()
        profile = self._opp_profile
        MIN_OBS = 5

        if facing_raise:
            # Facing our raise: fold / call / reraise
            pf_folds = sum(profile._folds_to_our_raise[1:4]) if profile else 0
            pf_calls = sum(profile._calls_to_our_raise[1:4]) if profile else 0
            total_faced = pf_folds + pf_calls

            if total_faced >= MIN_OBS:
                # Observed postflop fold/call split against our raises
                fold_rate = pf_folds / total_faced
                call_rate = pf_calls / total_faced
                # Scale down to leave room for re-raise estimate
                reraise_frac = min(0.15, self._opp_raise_rate * 0.25)
                scale = 1.0 - reraise_frac
                fold_rate *= scale
                call_rate *= scale
            else:
                fold_rate = self._opp_fold_rate
                reraise_frac = min(0.15, self._opp_raise_rate * 0.30)
                call_rate = max(0.0, 1.0 - fold_rate - reraise_frac)

            raw = {FOLD: fold_rate, CALL: call_rate}
            for ra in (RAISE_SMALL, RAISE_LARGE, RAISE_OVERBET, RAISE_ALLIN):
                raw[ra] = reraise_frac * raise_dist.get(ra, 0.0)
        else:
            # Not facing raise: check / bet(raise)
            pf_checks = sum(profile._checks[1:4]) if profile else 0
            pf_raises = sum(profile._raises[1:4]) if profile else 0
            total_open = pf_checks + pf_raises

            if total_open >= MIN_OBS:
                bet_rate = pf_raises / total_open
            else:
                bet_rate = self._opp_raise_rate

            raw = {FOLD: 0.0, CALL: 1.0 - bet_rate}
            for ra in (RAISE_SMALL, RAISE_LARGE, RAISE_OVERBET, RAISE_ALLIN):
                raw[ra] = bet_rate * raise_dist.get(ra, 0.0)

        sv = [max(0.0, raw.get(a, 0.0)) for a in actions]
        total = sum(sv)
        if total > 0:
            return [s / total for s in sv]
        return [1.0 / len(actions)] * len(actions)
