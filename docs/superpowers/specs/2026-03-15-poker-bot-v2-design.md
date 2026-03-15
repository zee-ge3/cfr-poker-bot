# Poker Bot v2 — Structural Overhaul Design Spec

## Goal

Build a new poker bot from scratch that:
- Is not exploitable (probabilistic decisions, balanced ranges, no hard cutoffs)
- Maximizes tournament win rate across all opponent types
- Uses precomputed exact equity (zero MC noise)
- Adapts dynamically to opponent tendencies and match phase
- Ships ASAP and supports continuous iteration

## Tournament Context

- **Format:** 1000-hand heads-up matches, stack resets to 100 each hand
- **Deck:** 27 cards (ranks 2-9, A; suits d, h, s). Card encoding: `suit * 9 + rank`
- **Hand flow:** Pre-flop (5 cards, betting) → Flop (3 community, discard to 2, betting) → Turn (4th card, betting) → River (5th card, betting, showdown)
- **Evaluator:** WrappedEval (handles Ace-as-10 for straights like 6-7-8-9-A)
- **Phases:** Phase 1 (500s, 1 vCPU, 2GB), Phase 2 (1000s, 2 vCPU, 4GB), Phase 3 (1500s, 4 vCPU, 8GB)
- **Submission:** `submission/player.py` with `class PlayerAgent(Agent)`. Multiple files allowed, < 1 GB total.

## Key Learnings from Tournament Analysis

Failure modes identified across 15+ match analyses of PA v18r and C2 v37:

1. **Post-flop fold-to-raise 80-86%** (C2): Call thresholds too high for MC equity distribution. Opponents exploit by min-raising every flop.
2. **Flop fold threshold ~0.62** (PA v18): Folding with average equity 0.482. Fixed in v18r but exposed new issues.
3. **Not raising enough vs folders** (PA v18r): Opponent folded 82% to post-flop raises, but PA only raised 266 times vs opponent's 1013. Massive +EV bluff raises left on the table.
4. **MC noise** (both): Equity estimates vary ±5-10% per call. Leads to inconsistent decisions and incorrect equity at showdown (reporting 0.81 equity on hands that lose).
5. **Opponent misclassification** (PA): "Alan Keating" (99% VPIP, 50% PF raise) classified as "tag" instead of "lag".
6. **No bluff rate integration** (C2): `_opp_bluff_rate()` defined but never called. Call threshold doesn't account for opponent bluff frequency.
7. **Pre-flop fold bleeding** (both): 40-50% PF fold rate against aggressive raisers leaks 400-900 chips per match.
8. **Hard threshold exploitation** (C2 v10): Deterministic fold/call boundaries allow opponents to predict behavior exactly.

## Architecture

```
submission/
├── player.py              # PlayerAgent — thin orchestrator
├── equity.py              # Exact equity engine (precomputed + weighted enumeration)
├── opponent.py            # Opponent model (range weights, tendencies, classification)
├── strategy.py            # Adaptive strategy (balanced baseline + exploit deviations)
├── match_manager.py       # Phase tracking, lock-in, time budget, hand-number dynamics
├── card_utils.py          # Card encoding, combinatorial indexing, hand classification
├── tables/
│   ├── hand_ranks.npy     # 7-card combo → hand score (888,030 entries, ~1.8 MB)
│   ├── hand_strength_2.npy # 2-card avg equity (351 entries, 1.4 KB)
│   └── hand_potential_5.npy # 5-card pre-flop potential (80,730 entries, 323 KB)
└── generate_tables.py     # Offline table generation script (NOT shipped in submission)
```

**Total precomputed data: ~2.1 MB** (well within 1 GB limit).

## Module 1: Card Utilities (`card_utils.py`)

Provides card encoding/decoding, combinatorial indexing, and hand classification.

### Card encoding

Game uses `card_int = suit_index * 9 + rank_index` where:
- rank_index: 0=2, 1=3, ..., 7=9, 8=Ace
- suit_index: 0=diamonds, 1=hearts, 2=spades

Helper functions:
- `rank(card) -> int`: `card % 9`
- `suit(card) -> int`: `card // 9`
- `is_suited(c1, c2) -> bool`: same suit
- `is_paired(c1, c2) -> bool`: same rank
- `is_connected(c1, c2) -> bool`: rank difference <= 2, OR ace-wrap (A-2, A-3 are connected since A-2-3-4-5 is a valid straight)
- `is_high(card) -> bool`: rank >= 6 (8, 9, or Ace)

### Combinatorial indexing

Maps sorted k-card tuples to unique indices using the combinatorial number system:

```python
# Precompute C(n, k) table for n=0..26, k=0..7
COMB = [[0] * 8 for _ in range(28)]
# ... fill with binomial coefficients

def combo_index(cards: tuple, k: int) -> int:
    """Map sorted k-card combo to unique index in [0, C(27,k))."""
    s = sorted(cards)
    return sum(COMB[s[i]][i + 1] for i in range(k))
```

Provides `combo_index_2`, `combo_index_5`, `combo_index_7` convenience wrappers.

### Hand classification

```python
def classify_hand(score: int) -> str:
    """Classify treys/WrappedEval score into hand type."""
    if score <= 10:    return "straight_flush"
    if score <= 322:   return "full_house"
    if score <= 1599:  return "flush"
    if score <= 1609:  return "straight"
    if score <= 2467:  return "trips"
    if score <= 3325:  return "two_pair"
    if score <= 6185:  return "pair"
    return "high_card"
```

## Module 2: Equity Engine (`equity.py`)

Provides exact equity computation using precomputed hand rank tables. Zero Monte Carlo sampling.

### Precomputed tables

**Table 1: `hand_ranks.npy`** — shape `(888_030,)`, dtype `uint16`
- Maps every 7-card combination (sorted) from the 27-card deck to its WrappedEval hand score
- Indexed via `combo_index_7(sorted(my_2 + board_5))`
- Lower score = stronger hand

**Table 2: `hand_strength_2.npy`** — shape `(351,)`, dtype `float32`
- Average equity of each 2-card hand across all possible boards and opponents (uniform random)
- Indexed via `combo_index_2(sorted(hand))`
- Used for quick pre-flop reference and opponent weighting

**Table 3: `hand_potential_5.npy`** — shape `(80_730,)`, dtype `float32`
- Expected equity of each 5-card starting hand when played optimally (best discard chosen after seeing flop)
- Computed offline by sampling ~200 flops per hand, finding optimal discard, averaging equity
- Indexed via `combo_index_5(sorted(hand))`
- Used for pre-flop call/raise decisions

### Core functions

```python
def hand_rank(my_2: tuple, board_5: tuple) -> int:
    """Exact hand score from precomputed table. O(1)."""
    return HAND_RANKS[combo_index_7(sorted(my_2 + board_5))]

def exact_equity(my_2, board, dead_cards, opp_weights_fn=None) -> float:
    """
    Exact equity via enumeration of all possible opponent hands
    and remaining board cards, weighted by opponent preferences.

    Args:
        my_2: our 2 kept cards
        board: known community cards (3-5)
        dead_cards: known dead cards (our discards + opponent discards)
        opp_weights_fn: function(opp_2_card_tuple) → float weight

    Returns: win probability [0.0, 1.0]
    """

def optimal_discard(hole_5, flop_3, opp_weights_fn=None) -> tuple[int, int]:
    """
    Evaluate all C(5,2)=10 keep options with exact equity.
    Returns (index_i, index_j) of the two cards to keep.
    """

def preflop_strength(hole_5: tuple) -> float:
    """Instant lookup of 5-card pre-flop potential."""
    return HAND_POTENTIAL_5[combo_index_5(sorted(hole_5))]
```

### Performance

| Operation | Lookups | Time |
|-----------|---------|------|
| River equity (5 board) | ~91-136 | <0.1ms |
| Turn equity (4 board) | ~1,200-1,400 | <0.5ms |
| Flop equity (3 board) | ~10,000-11,000 | <2ms |
| Optimal discard (10 × flop equity) | ~100,000-110,000 | <15ms |
| Pre-flop strength | 1 | <0.01ms |
| **Total worst case per hand** | | **<20ms** |

Compared to current MC: 50-200ms per call, multiple calls per hand, noisy. New engine is 10-100x faster AND exact.

## Module 3: Opponent Model (`opponent.py`)

Observes all opponent actions and exposes a clean context dict. Never makes decisions.

### State tracked

Per-street action counts (raise, call, check, fold counts for each of 4 streets), bet sizing history (last 200 raise amounts as fraction of pot), per-hand action sequence (for trap/barrel detection), range preference weights (suited, paired, connected, high — learned from discards and showdowns), showdown calibration (hand class correlated with raise behavior per street), bluff rate (Bayesian Beta distribution), fold response to our raises.

### Key interface

```python
def get_context(self, street: int) -> dict:
    """Returns all opponent intelligence for strategy module."""
    # Returns: raise_rate, raise_rate_by_street, raises_often,
    # fold_rate_to_our_raise, fold_rate_pf, folds_often,
    # bluff_rate, raise_strength, avg_raise_frac,
    # current_is_small, current_is_large,
    # is_trap_line, is_barrel_line,
    # opp_weights_fn, opponent_type

def update_action(self, action, street, bet_size, pot): ...
def update_discards(self, opp_discards): ...
def update_showdown(self, opp_hand, hand_score, raised_streets): ...
def reset_hand(self): ...
```

### Range weighting function

```python
def opp_weights_fn(self, hand: tuple) -> float:
    """Weight for a possible opponent 2-card holding based on learned preferences."""
    c1, c2 = hand
    s1, r1 = c1 // 9, c1 % 9
    s2, r2 = c2 // 9, c2 % 9
    w = 1.0
    if s1 == s2:             w *= self.pref_suited     # suited
    if r1 == r2:             w *= self.pref_paired      # paired
    if abs(r1 - r2) <= 2:    w *= self.pref_connected   # connected
    if r1 >= 6 or r2 >= 6:   w *= self.pref_high        # high cards
    return w
```

All preferences start at 1.0 (neutral). Updated from discards (every hand) and showdowns (when visible).

### Bluff rate estimation

Bayesian Beta distribution: prior Beta(2, 8) = 20% bluff rate.
- At showdown after opponent raised: strong hand → beta += 1, weak hand → alpha += 1.
- Estimate: alpha / (alpha + beta). Converges to true rate, no hardcoded caps.

### Opponent type classification

Based on raise rate AND fold-to-our-raise (fixes PA's misclassification bug):
- LAG: raises often (>0.40), doesn't fold to resistance (<0.40)
- TAG: raises often (>0.40), folds to resistance (>=0.40)
- Station: passive (<=0.40), doesn't fold (<0.40)
- Nit: passive (<=0.40), folds often (>=0.40)

## Module 4: Strategy (`strategy.py`)

The decision engine. Takes equity + opponent context + match state, returns an action. Every decision uses sigmoid probabilistic thresholds — no hard cutoffs anywhere.

### Sigmoid threshold

```python
def soft_decision(value, threshold, temp=0.04) -> float:
    """P(action) = sigmoid((value - threshold) / temp)."""
    x = (value - threshold) / temp
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
```

### Decision flow

1. **Lock-in check:** If `in_lockout` → auto-fold. If `in_protection` and equity < 0.60 → fold/check.

2. **Pre-flop decisions:** Based on `preflop_strength(5 cards)`.
   - Call threshold: `pot_odds + 0.05`, adjusted down by opponent PF raise rate and early aggression.
   - Floor: 0.15 (absolute minimum). All through `soft_decision()`.

3. **Post-flop — three layers:**

**Layer A: Should I raise? (Aggression)**

Value raises: equity above value threshold (0.62 base, lowered vs folders).

Bluff raises (THE key new feature): When `fold_rate_to_our_raise * pot > (1 - fold_rate) * raise_cost`, raising is +EV regardless of hand strength. Bluff probability proportional to profitability, capped at 0.40 for balance. Minimum equity: 0.10 (flop/turn), 0.20 (river). Bluff sizing: 0.50x pot (cheaper). Value sizing: 0.75x pot (standard). All with ±15% jitter.

**Layer B: Should I call their raise? (Defense)**

```
call_thr = (pot_odds + 0.03) × (1 - bluff_rate)
```

This directly integrates bluff rate. At 40% bluff rate: threshold drops from 0.28 to 0.17. Most post-flop hands (0.30-0.45 equity) now call instead of fold. Fixes the 80% fold-to-raise problem.

Additional modifiers: sizing tell (±0.03), action line (trap +0.05, barrel -0.04), early aggression (-0.04). Floor: 0.10 (never fold getting >5:1 with any equity). All through `soft_decision()`.

**Layer C: Should I probe? (Initiative)**

When checked to us: probe threshold 0.45 base, drops to 0.20 vs folders. Through `soft_decision()`.

**Raise war cap:** If opponent re-raises our raise, require equity 0.65+ to continue (0.55 vs high bluff rate opponents).

### Raise sizing

- Bluff: 0.50x pot (minimize risk)
- Value: 0.75x pot (standard)
- Reduced vs folders: 0.60x pot (keep them calling)
- All with ±15% jitter to prevent sizing tells

## Module 5: Match Manager (`match_manager.py`)

Tracks match state and provides continuous (not discrete) phase signals.

### State

```python
hand_number: int          # 0-999
cumulative_reward: float  # running chip total
total_hands: int          # 1000
time_left: float          # seconds remaining
phase_mult: int           # 1 (500s), 2 (1000s), or 3 (1500s)
```

### Computed signals

- **pressure** [-1, +1]: -1 = comfortable lead, +1 = desperate. Based on cumulative reward relative to remaining hands × bleed rate.
- **aggression** [0, 1]: Early game factor. `max(0, 1 - hand/300) × (1 + uniform(-0.20, 0.20))`. High early, decays smoothly. Suppressed in protection mode. Combined with desperation: `max(early_factor, max(0, pressure) × 0.8)`.
- **in_protection** [bool]: Lead > 15% of remaining bleed → minimize variance.
- **in_lockout** [bool]: Lead > remaining_hands × 1.5 + buffer → auto-fold everything.
- **time_per_hand** [float]: Adaptive. With exact equity taking <20ms/hand, time is no longer a constraint. But still tracked for safety.

All transitions are smooth and jittered. No hard phase boundaries detectable by opponents.

## Module 6: Player Orchestrator (`player.py`)

Thin wiring layer. Inherits from `Agent`, implements `act()` and `observe()`.

### act() flow

```
1. match_manager.check_lockout() → auto-fold if applicable
2. If street changed: reset per-street state, update opponent model with opp_last_action
3. If DISCARD required (street 1, first action):
     equity.optimal_discard(hole_5, flop_3, opponent.opp_weights_fn) → keep indices
4. If BETTING:
     eq = equity.exact_equity(my_2, board, dead_cards, opponent.opp_weights_fn)
     opp_ctx = opponent.get_context(street)
     match_state = match_manager.get_state()
     action = strategy.decide(eq, pot_odds, street, ..., opp_ctx, match_state, obs)
5. Track our action (for raise war detection, bet recording)
6. Return action tuple
```

### observe() flow

```
1. If terminated:
     match_manager.update(reward)
     opponent.reset_hand()
     If showdown: check "player_0_cards" in info dict (only present when
       street > 3, i.e. hand reached showdown without a fold).
       Parse opponent's kept cards, evaluate, update opponent model.
     opponent.update_showdown(opp_hand, score, raised_streets)
2. Update opponent model with terminal action
   Note: opp_last_action is injected by the match runner into observations,
   not by the engine directly. It contains strings like "RAISE_50", "CALL",
   "CHECK", "FOLD", "DISCARD".
```

## Table Generation (`generate_tables.py`)

Offline script, NOT shipped in submission. Generates all three `.npy` files.

### hand_ranks.npy

```python
evaluator = PokerEnv().evaluator
table = np.zeros(888_030, dtype=np.uint16)
for combo in combinations(range(27), 7):
    # Any 2+5 split gives the same score (partition-independent, verified)
    cards_2 = [int_to_treys(combo[0]), int_to_treys(combo[1])]
    board_5 = [int_to_treys(c) for c in combo[2:]]
    score = evaluator.evaluate(cards_2, board_5)
    table[combo_index_7(combo)] = score
```

**Partition independence (verified):** `WrappedEval.evaluate(hand_2, board_5)` finds the best 5-card hand from all 7 cards regardless of which 2 are designated as "hand." All 21 possible 2+5 splits of the same 7 cards produce identical scores, including the Ace-as-10 edge case (6-7-8-9-A straight). The table stores one score per 7-card set, and any caller-provided 2+5 split will get the correct result.

**Validation anchors:** During generation, verify:
- `combo_index_7((0,1,2,3,4,5,6))` returns `0`
- `combo_index_7((20,21,22,23,24,25,26))` returns `888029`

### hand_strength_2.npy

For each 2-card hand, enumerate all boards and opponents using the hand_ranks table, compute average win rate. 351 entries. With table lookups (~100M/sec), generation takes ~35 seconds.

### hand_potential_5.npy

For each 5-card hand, sample 200 random flops, find optimal discard for each, average the best equity. 80,730 entries, feasible in ~2-4 hours with the hand_ranks table available.

**Note:** This table uses sampling (200 flops), so pre-flop decisions carry ~3-5% noise. Only post-flop decisions (with known board cards) are truly exact. This is acceptable because pre-flop decisions are lower-stakes (1-2 chip blinds) and the noise is far smaller than MC sampling (~10-15% noise). If needed, can increase to 500+ samples or compute exactly in later iterations.

## Cross-Match Learning & Refinement

The bot improves between submissions via offline log analysis and tunable parameters.

### Tunable parameters (`tables/strategy_params.json`)

All strategy thresholds load from a config file rather than being hardcoded:

```json
{
  "pf_call_margin": 0.05,
  "call_thr_base_margin": 0.03,
  "bluff_cap": 0.40,
  "bluff_min_equity_flop": 0.10,
  "bluff_min_equity_river": 0.20,
  "value_raise_thr": 0.62,
  "probe_thr_base": 0.45,
  "probe_thr_vs_folder": 0.20,
  "reraise_thr": 0.65,
  "protection_equity_min": 0.60,
  "bluff_prior_alpha": 2,
  "bluff_prior_beta": 8,
  "early_aggression_decay": 300
}
```

After each tournament round, log analysis identifies which thresholds cost chips → update params → resubmit.

### Opponent priors (`tables/opponent_priors.json`)

Pre-loaded profiles for known opponents. Gives 10-20 hand calibration head start:

```json
{
  "AlbertLuoLovers": {
    "pf_raise_rate": 0.37,
    "pf_raise_size": 20,
    "bluff_prior": [4, 6],
    "fold_rate_postflop": 0.12,
    "pref_suited": 1.3,
    "updated_match_id": 5633
  }
}
```

At match start, check opponent name from observation/info. If prior exists, initialize opponent model with those weights instead of neutral defaults.

**Critical: opponent evolution handling.** Opponents also refine their bots between matches. Stale priors are dangerous. Two safeguards:

1. **Confidence decay:** Prior data weighted by recency. Each prior carries `updated_match_id`. Older priors decay: weight = `0.8 ^ (matches_since_update)`. After ~5 matches without update, prior weight is negligible.

2. **Drift detection:** If observed behavior diverges significantly from prior within the first 15-30 hands, reset to neutral:
   ```python
   if abs(observed_rate - prior_rate) > 0.25 and hands_observed > 15:
       self.prior_weight = 0.0  # opponent changed, discard stale prior
   ```
   This ensures priors help when opponents are stable but never lock us into a wrong read.

3. **Priors are soft, not hard:** Prior values blend with observations via exponential moving average, not replace them. By hand 50, observations dominate regardless of prior.

### Offline analysis script (`analyze_logs.py`, NOT shipped)

Processes tournament game logs and outputs:

1. **Threshold calibration:** Identify folds where equity > call threshold → quantify chips left on table → suggest threshold adjustments.
2. **Bluff efficiency:** Track bluff raise success rate → adjust bluff_cap and min_equity params.
3. **Opponent scouting:** Build/update `opponent_priors.json` from observed tendencies.
4. **Discard quality audit:** Compare our discards against optimal → flag systematic errors.
5. **Strategy regression detection:** Compare current EV/hand against previous versions.

### Refinement cycle

```
Tournament round N:
  1. Matches play out → logs collected
  2. analyze_logs.py processes results
  3. Update strategy_params.json (threshold tuning)
  4. Update opponent_priors.json (scouting, with recency decay)
  5. Optionally: refine hand_potential_5.npy with more samples
  6. Resubmit → tournament round N+1
```

## Testing Strategy

### Unit tests
1. **Table verification:** Generate tables, spot-check against WrappedEval for random hands.
2. **Equity accuracy:** Compare exact_equity output against high-sample MC for 100 random situations. Must match within 0.001.
3. **Discard optimality:** For 100 random (5-card, flop) situations, verify optimal_discard matches exhaustive MC search.

### Integration tests
4. **Lock-in test:** Verify auto-fold activates at correct lead thresholds and match completes without timeout.
5. **Exploit verification:** Against a "raise every flop" opponent, verify fold-to-raise rate < 40% (was 80%+ before).
6. **Balance verification:** Against a "fold everything" opponent, verify raise frequency increases proportionally.

### Local round-robin simulation
7. **Round-robin vs PA v18r and C2 v37:** Run the new bot against both existing bots using the local match runner. Minimum 10 matches per pairing (10,000 hands each). Measure:
   - EV/hand (must be positive against both)
   - Post-flop fold-to-raise rate (must be < 50% against aggressive opponents)
   - Bluff raise frequency (must increase proportionally with opponent fold rate)
   - Showdown win rate (must be > 50%)
   - Time per hand (must average < 50ms with headroom for phase 1's 500s budget)
8. **Regression check:** New bot must not lose to either PA or C2 by more than their historical loss margins against tournament opponents.

### Pre-submit validation
9. **Full 1000-hand match timing:** Verify total time < 400s (Phase 1), < 800s (Phase 2), < 1200s (Phase 3) with margin.
10. **Submission packaging:** Verify `from submission.player import PlayerAgent` works, all tables load correctly, no missing imports.

## Implementation Priority

1. `card_utils.py` + `generate_tables.py` → generate and verify tables
2. `equity.py` → exact equity engine, verify against MC
3. `match_manager.py` → phase/pressure/lock-in logic
4. `opponent.py` → opponent model with range weighting
5. `strategy.py` → adaptive strategy with bluff-rate-adjusted calls + EV bluff raises
6. `player.py` → wire everything together
7. Local round-robin testing vs PA v18r and C2 v37
8. Submit → analyze logs → refine params → resubmit

## Risk Mitigation

- **WrappedEval partition assumption:** Verified — evaluate(hand, board) gives same result regardless of 2+5 split.
- **Table generation time:** hand_potential_5.npy takes hours. Can ship v1 without it (use hand_strength_2 for pre-flop) and add it in v2.
- **New bugs in fresh code:** Mitigated by modular design (each module testable independently), local round-robin testing against PA/C2, and integration tests.
- **Stale opponent priors:** Handled by confidence decay and drift detection. Priors help but never override live observations.
- **Numpy dependency:** All table operations use numpy which is pre-installed. No external dependency risk.
- **ARM64 portability:** Tournament runs on AWS Graviton2 (ARM64). Numpy `.npy` files use platform-independent format with explicit endianness — tables generated on x86 are fully portable.
- **Pre-flop noise:** `hand_potential_5.npy` uses 200-sample estimation, introducing ~3-5% noise in pre-flop decisions. Acceptable for low-stakes pre-flop bets (1-2 chips). Can increase samples or compute exactly in later versions.
