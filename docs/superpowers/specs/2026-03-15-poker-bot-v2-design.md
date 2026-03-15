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
- Computed offline by exhaustive enumeration of ALL C(22,3) = 1,540 possible flops per hand (no sampling)
- For each (hand, flop): find optimal keep using **flop-aware exact_equity** (NOT hand_strength_2)
- **Critical:** Using hand_strength_2 (average equity across all boards) to evaluate keeps would commit
  the same "post-flop blind" error as the rationality filter. Example: hand [Ad,Kd,9s,9h,2c] on flop
  3d-4d-8d — hand_strength_2 ranks 9s-9h (pair) above Ad-Kd (unpaired), but any rational player keeps
  Ad-Kd for the nut flush draw. The offline generator must call `exact_equity(keep, flop, discard)` for
  each of the 10 keeps per flop.
- 80,730 × 1,540 × 10 keeps × ~10,920 equity lookups each = ~13.6T total lookups
- This is computationally expensive (~hours) but runs OFFLINE once. Can be parallelized across CPU cores.
- Indexed via `combo_index_5(sorted(hand))`
- Used for pre-flop call/raise decisions
- **Truly exact** — no sampling noise, no MC approximation, no pre-flop-average shortcuts

### Core functions

```python
def hand_rank(my_2: tuple, board_5: tuple) -> int:
    """Exact hand score from precomputed table. O(1)."""
    return HAND_RANKS[combo_index_7(sorted(my_2 + board_5))]

def exact_equity(my_2, board, dead_cards, opp_discards=None,
                 my_discards=None, opp_weights_fn=None) -> float:
    """
    Exact equity via enumeration of RATIONAL opponent hands only.

    Critical: opponents were dealt 5 cards and kept their best 2. Naive
    enumeration of all C(remaining, 2) treats them as if dealt 2 cards,
    overestimating our equity because it includes garbage hands they would
    never keep. We must filter: for each candidate opponent hand (a,b),
    verify that (a,b) is the rational keep from {a,b} + opp_discards.

    Args:
        my_2: our 2 kept cards
        board: known community cards (3-5)
        dead_cards: known dead cards (our discards + opponent discards)
        opp_discards: opponent's 3 revealed discards (for rationality filter)
        my_discards: our 3 discards (opponent sees these before choosing their keep —
                     must be treated as dead cards in the rationality filter's keep
                     evaluation, otherwise we assign weight to flush/straight draws
                     the opponent would never keep because they saw us discard their outs)
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

### Rationality filter (best-2-from-5 correction, FLOP-AWARE)

Opponents are dealt 5 cards and keep their best 2 AFTER seeing the flop. Naive enumeration
of C(remaining, 2) treats them as if dealt 2 random cards — this overestimates our equity
because it includes hands the opponent would never keep.

**Critical:** The filter MUST evaluate keep options against the actual flop, not pre-flop
averages. Example: flop is 3d-4d-8d. Opponent was dealt [Ad,Kd,9s,9h,2c]. Pre-flop
averages score 9s-9h (pair) higher than Ad-Kd (unpaired). But any rational player keeps
Ad-Kd for the nut flush draw on a three-diamond flop. Using pre-flop averages would
filter OUT the flush draw and produce catastrophically wrong equity.

**Fix:** Evaluate each keep option's equity against the known flop using `exact_equity()`.
This is called ONCE per hand (when building the filtered opponent range), not per-street:

**Conceptual logic (DO NOT implement as nested Python loops):**

```python
def rational_opponent_hands(remaining, opp_discards, flop, my_discards):
    """Yield only hands the opponent would rationally keep given the flop."""
    # dead_for_opp_eval = flop + my_discards (opponent sees our discards before deciding)
    dead_for_opp_eval = list(flop) + list(my_discards)
    for hand in combinations(remaining, 2):
        original_5 = list(hand) + list(opp_discards)
        best_eq = -1
        best_keep = None
        for i, j in combinations(range(5), 2):
            keep = (original_5[i], original_5[j])
            discard = [original_5[k] for k in range(5) if k not in (i, j)]
            # Flop-aware equity with MY discards as dead cards
            eq = exact_equity(keep, flop, discard + dead_for_opp_eval)
            if eq > best_eq:
                best_eq = eq
                best_keep = set(keep)
        if best_keep == set(hand):
            yield hand
```

**Critical: our discards are dead cards for the opponent's keep evaluation.** The opponent
sees our 3 discards before choosing their keep. If we discarded 3 hearts, the opponent
knows those hearts are dead — making heart flush draws weaker for them. Omitting our
discards from the filter would assign weight to opponent flush/straight draws they would
never actually keep because they saw us discard their outs.

**MANDATORY: Vectorized implementation.** The above Python loops are for clarity only.
Nested `for` loops in Python incur ~50-100ns overhead per iteration. 13M Python-level
iterations would take 1-2 seconds, blowing past Phase 1's 0.5s/hand budget. The actual
implementation MUST use NumPy vectorization:

```python
def rational_opponent_hands_vectorized(remaining, opp_discards, flop, my_discards):
    """Vectorized rationality filter using batch table lookups."""
    # 1. Pre-generate ALL candidate (hand, keep) combos as index arrays
    remaining = np.array(remaining, dtype=np.int32)
    opp_disc = np.array(opp_discards, dtype=np.int32)
    dead = np.concatenate([np.array(flop), np.array(my_discards)])

    # 2. All C(remaining, 2) candidate hands → shape (N, 2)
    cand_hands = np.array(list(combinations(remaining, 2)), dtype=np.int32)  # ~120 rows

    # 3. For each candidate hand, reconstruct original_5 = hand + opp_discards
    #    All 10 keep options are fixed index pairs into the 5-card array
    KEEP_INDICES = np.array(list(combinations(range(5), 2)), dtype=np.int32)  # (10, 2)

    # 4. Build all original_5 arrays: shape (N, 5)
    orig5 = np.concatenate([cand_hands, np.tile(opp_disc, (len(cand_hands), 1))], axis=1)

    # 5. Extract all keeps: shape (N, 10, 2) and all discards: shape (N, 10, 3)
    all_keeps = orig5[:, KEEP_INDICES]  # fancy indexing

    # 6. For each (N×10) keep, compute exact_equity via BATCH table lookup
    #    This is the inner hot loop — must be a single vectorized operation
    #    Pre-build all 7-card combos (keep + board + remaining_cards),
    #    index into HAND_RANKS table, compute win fractions via broadcasting
    equities = batch_exact_equity(all_keeps, flop, dead)  # shape (N, 10)

    # 7. Best keep per candidate = argmax along axis 1
    best_keep_idx = np.argmax(equities, axis=1)  # shape (N,)

    # 8. Filter: keep only candidates where their actual hand IS the best keep
    #    Candidate hand = indices (0,1) in original_5 → keep_idx 0
    mask = (best_keep_idx == 0)  # keep_indices[0] = (0, 1) = the candidate hand itself
    return cand_hands[mask]
```

**`batch_exact_equity()`** is the critical function: it takes all (N×10) keeps, generates
ALL opponent board completions as index arrays, does a single `HAND_RANKS[indices]` lookup
(pure NumPy C-level), and computes win rates via vectorized comparison. Zero Python loops
in the hot path.

**Realistic performance estimate (vectorized):**
- Pre-generate index arrays: ~0.5ms (Python overhead, one-time)
- Batch table lookups: 13M entries × ~10ns/lookup (NumPy vectorized) = ~130ms
- Total: ~130-150ms per hand (one-time, cached)

**Two-tier optimization for Phase 1 (500s budget, 0.5s/hand):**
1. Fast pre-filter: use `hand_strength_2` to eliminate keeps that are obviously worse than
   the best keep by >0.15 equity → reduces candidates from ~120 to ~50
2. Vectorized flop-aware filter on remaining ~50 → ~5.5M lookups = ~55-70ms

**Fallback:** If even vectorized is too slow, pre-filter aggressively (keep only top-3
keeps per candidate by hand_strength_2) → 120 × 3 × 10,920 = ~3.9M lookups = ~40ms.
Accuracy cost is minimal — the pre-filter only discards keeps that are bad on average
AND bad on this specific flop.

**Impact:** Correctly handles flush draws, straight draws, and situational keeps that
pre-flop averages would miss. Typically filters from ~120 candidates down to ~30-50
consistent hands, with the RIGHT hands surviving (flush draws on flush boards, pairs
on dry boards).

### Performance (vectorized NumPy — realistic estimates)

**Critical:** All time estimates assume vectorized NumPy operations (batch array indexing,
broadcasting). Pure Python for-loop implementations would be 10-20x slower and are NOT
acceptable. Every equity computation must hit the `HAND_RANKS` table via NumPy fancy
indexing, not Python iteration.

| Operation | Lookups | Time (vectorized) |
|-----------|---------|------|
| River equity (5 board) | ~50-80 (filtered) | <0.5ms |
| Turn equity (4 board) | ~700-1,000 (filtered) | <2ms |
| Flop equity (3 board) | ~5,000-8,000 (filtered) | <5ms |
| Optimal discard (10 × flop equity) | ~50,000-80,000 | <30ms |
| Pre-flop strength | 1 | <0.01ms |
| Rationality filter (vectorized, one-time) | ~5.5M-13M | ~55-150ms |
| **Total worst case per hand** | | **<200ms** |

Phase 1 budget: 500s / 1000 hands = 500ms/hand. At ~200ms worst case, we use ~40% of
budget with comfortable margin. Phase 2/3 have 2-3x more time.

Compared to current MC: 50-200ms per call, multiple calls per hand, noisy. New engine
is comparable speed but EXACT — no variance, no re-rolling. The rationality filter is
a one-time cost that actually reduces subsequent equity calls by eliminating impossible
opponent hands.

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

Bluff raises (THE key new feature): Sizing is determined FIRST by `bet_size(pot, board, persona)`
(board-texture-based, see Defense 4). Then the ACTUAL raise_cost is plugged into the EV formula:

```python
raise_amount = bet_size(pot, board, persona)  # board-texture sizing, NOT hand-dependent
raise_cost = raise_amount
total_pot_if_called = pot + 2 * raise_cost  # our raise + their call
bluff_ev = (fold_rate * pot) + ((1 - fold_rate) * (equity * total_pot_if_called - raise_cost))
```

**Why equity matters when called:** The naive formula `fold_rate × pot - (1 - fold_rate) × raise_cost`
assumes equity = 0 when called (we always lose the raise_cost). In reality, our "bluffs" often
have 0.10-0.30 equity (semi-bluffs: flush draws, overcards, gutshots). When called, we still win
`equity × (pot + 2 × raise_cost)` at showdown. Omitting this term makes the bot severely
under-bluff, evaluating profitable semi-bluffs as negative EV.

Example: pot=20, raise_cost=10, fold_rate=0.50, equity=0.25.
- Naive formula: 0.50 × 20 - 0.50 × 10 = +5.0 (correct sign, wrong magnitude)
- Full formula: (0.50 × 20) + (0.50 × (0.25 × 40 - 10)) = 10 + 0 = +10.0
- With equity=0.10: (0.50 × 20) + (0.50 × (0.10 × 40 - 10)) = 10 + (-3) = +7.0
- The full formula captures semi-bluff value that the naive formula misses entirely.

**GTO-anchored bluff frequency (anti-adaptation safeguard):**

Pure exploitative bluffing (`if bluff_ev > 0: bluff`) is vulnerable to responsive opponents.
If they notice we're over-bluffing, their fold_rate drops to near zero. Our Bayesian fold_rate
estimate lags by 10-20 hands, during which we hemorrhage chips bluffing into a calling station.

The fix is a GTO bluff ceiling derived from our bet sizing. When betting `b` into pot `p`,
the opponent needs equity `b / (p + 2b)` to call. To make them indifferent, our range must
contain `b / (p + b)` BLUFF-TO-VALUE ratio, which translates to a bluff *frequency* of
`b / (p + 2b)` of our total betting range:

```python
bet_frac = raise_cost / max(1, pot)
gto_bluff_ratio = bet_frac / (1.0 + 2 * bet_frac)  # e.g., pot-sized bet → 33% bluffs
# NOT bet_frac / (1 + bet_frac) — that is the MDF for the CALLER, not the bettor's bluff ratio
# Pot-sized: 1/(1+2*1) = 33%. Half-pot: 0.5/(1+1) = 25%. 0.75x pot: 0.75/2.5 = 30%.
```

The actual bluff probability is the MINIMUM of the exploitative and GTO ceilings:

```python
exploit_bluff_prob = min(1.0, bluff_ev / max(1, pot)) * persona["bluff_cap"]
gto_bluff_ceiling = gto_bluff_ratio  # from sizing math above

# Blend: lean exploitative when confidence is high, lean GTO when uncertain
confidence = min(1.0, fold_observations / 30)  # 0-1 based on data quality
bluff_prob = exploit_bluff_prob * confidence + gto_bluff_ceiling * (1 - confidence)
bluff_prob = min(bluff_prob, gto_bluff_ceiling * 1.3)  # never exceed GTO by >30%
```

This ensures:
- With few observations (hands 0-30): bluff frequency is near-GTO balanced
- With many observations (hands 30+): we can deviate up to 30% above GTO to exploit folders
- Against an opponent who adapts and stops folding: our bluff frequency naturally drops
  because (a) fold_rate decreases → bluff_ev drops, and (b) GTO ceiling caps us regardless
- We NEVER bluff at a rate that makes calling always profitable for the opponent

Minimum equity: 0.10 (flop/turn), 0.20 (river).

**Critical:** There is ONE sizing function (`bet_size`), used for BOTH bluffs and value.
It outputs the same size for the same board regardless of hand strength. No separate
bluff/value sizing — the counter-intelligence layer (Defense 4) and the strategy layer
are unified, not contradictory.

**Layer B: Should I call their raise? (Defense)**

```
call_thr = (pot_odds + 0.03) × (1 - bluff_rate)
```

This directly integrates bluff rate. At 40% bluff rate: threshold drops from 0.28 to 0.17. Most post-flop hands (0.30-0.45 equity) now call instead of fold. Fixes the 80% fold-to-raise problem.

Additional modifiers: sizing tell (±0.03), action line (trap +0.05, barrel -0.04), early aggression (-0.04). Floor: 0.10 (never fold getting >5:1 with any equity). All through `soft_decision()`.

**Layer C: Should I probe? (Initiative)**

When checked to us: probe threshold 0.45 base, drops to 0.20 vs folders. Through `soft_decision()`.

**Raise war cap:** If opponent re-raises our raise, require equity 0.65+ to continue (0.55 vs high bluff rate opponents).

### Raise sizing (unified, board-texture-based, range-balanced)

There is ONE sizing function used for ALL raises (value, bluff, probe). Sizing is
determined by board texture and per-match persona, NOT by hand strength. See Defense 4
for the full `bet_size()` implementation.

Key properties:
- Same board → same size regardless of whether we hold the nuts or air
- Wet boards → larger bets (0.70-0.95x pot). Dry boards → smaller bets (0.40-0.65x pot)
- Per-match persona shifts the center ±0.10x, preventing cross-match clustering
- ±20% jitter on top of everything
- Zero correlation between bet size and hand strength → immune to clustering attacks

**Range balance enforcement:** Board-texture sizing alone is insufficient if the FREQUENCY
of betting varies wildly between value and air across board types. If an opponent logs
"on wet boards, bot bets 0.95x pot — and 80% of the time it's a bluff," the sizing mask
is broken. Sizing dictates the required value-to-bluff ratio:

```python
# After bet_size() determines the raise amount:
bet_frac = raise_amount / max(1, pot)
gto_value_ratio = 1.0 - bet_frac / (1.0 + 2 * bet_frac)  # e.g., 0.75x → 70% value, pot-sized → 67%
```

The strategy module tracks its own betting frequency per board-texture class (wet/medium/dry)
and enforces that the bluff-to-value ratio stays within ±15% of the GTO ratio. If we've
been bluffing too much on wet boards, the next bluff opportunity on a wet board is suppressed
even if exploitatively +EV. This prevents opponents from profiling our range composition
per board type.

## Counter-Intelligence Layer (integrated across modules)

Opponents have access to match logs from games they played against us. With matches every
5 minutes over 6 days, a single opponent could accumulate 50+ matches of data. An opponent
running offline analysis (bet size clustering, discard deduction, threshold mapping) could
reverse-engineer our strategy completely if we play deterministically.

Every source of information leakage must be poisoned while keeping EV cost near zero.

### Defense 1: Discard Stochasticity (equity.py)

**Threat:** Our 3 discards are revealed. If `optimal_discard()` is deterministic, opponents
can run the same math on our discards and deduce our exact hole cards.

**Fix:** In `optimal_discard()`, after evaluating all 10 keeps, if the top-2 keeps are within
0.03 equity of each other, select the #2 keep with 20% probability:

```python
def optimal_discard(hole_5, flop_3, opp_weights_fn=None):
    keeps = []
    for i, j in combinations(range(5), 2):
        keep = (hole_5[i], hole_5[j])
        eq = exact_equity(keep, flop_3, ...)
        keeps.append((eq, i, j))
    keeps.sort(reverse=True)

    best_eq, bi, bj = keeps[0]
    second_eq, si, sj = keeps[1]

    # Stochastic mixer: if close, sometimes take #2 to hide information
    if best_eq - second_eq < 0.03 and random.random() < 0.20:
        return (si, sj)
    return (bi, bj)
```

**EV cost:** <0.006 per hand on average (0.20 × 0.03 equity × rare occurrence). Negligible.
**Information cost to opponent:** Cannot deduce our hole cards with certainty. Their
reverse-engineering scripts produce multiple plausible holdings instead of one.

### Defense 2: Per-Match Persona Rotation (match_manager.py)

**Threat:** If we play identical parameters every match, opponents build accurate priors
from past matches and counter us from hand 1.

**Fix:** At `__init__()`, randomize key strategy parameters within defined ranges:

```python
# In match_manager or strategy __init__:
self.persona = {
    "value_raise_thr": random.uniform(0.58, 0.66),
    "bluff_cap": random.uniform(0.25, 0.50),
    "pf_call_margin": random.uniform(0.03, 0.07),
    "probe_thr_base": random.uniform(0.40, 0.50),
    "call_thr_margin": random.uniform(0.02, 0.05),
    "bluff_size_center": random.uniform(0.55, 0.65),
    "value_size_center": random.uniform(0.70, 0.80),
}
```

Each match, the bot has a slightly different mathematical personality. An opponent analyzing
5 matches will see 5 different raise rates, 5 different sizing profiles, and 5 different
call thresholds. Their aggregate model will be noisy and unreliable.

**EV cost:** Near zero — all parameter ranges are within the profitable zone. The variance
in personality is smaller than the variance in card distribution.

### Defense 3: Showdown Data Poisoning (strategy.py)

**Threat:** Showdowns reveal ground truth about our range. If we only show down premium
hands, opponents map our exact call/raise thresholds.

**Fix:** Distribute poison across ALL pot sizes AND action lines so opponents cannot filter
by pot size or by "was a raise involved." Two poison vectors:

**Vector A — Passive pots (checked-to or limped):**
```python
# In strategy.decide(), after normal decision computed:
if street >= 2 and not facing_raise:
    poison_prob = 0.03 * math.exp(-pot / 15.0)
    if random.random() < poison_prob and equity < 0.35:
        return call() if can_call else check()
```

**Vector B — Raised pots (we were the aggressor), with `_poison_active` flag:**

Vector B requires a dedicated state flag because simply calling `check()` doesn't guarantee
the hand reaches showdown — if the opponent bets, standard strategy folds equity < 0.30,
defeating the entire purpose.

```python
# In player.py __init__:
self._poison_active = False  # persists across streets within a hand

# Trigger: when we raised earlier and equity is low
if street >= 2 and we_raised_earlier and not self._poison_active:
    poison_prob = 0.015 * math.exp(-pot / 20.0)
    if random.random() < poison_prob and equity < 0.30:
        self._poison_active = True  # FLAG THE HAND

# On EVERY subsequent decision in the same hand:
if self._poison_active:
    if facing_raise:
        return call()   # check-CALL even facing bets — forced showdown
    else:
        return check()  # check down when not facing bet
    # This overrides standard strategy for the rest of the hand

# Reset at hand end:
# In _reset_hand_state(): self._poison_active = False
```

**Why the flag is critical:** Without it, `check()` only works if the opponent also checks.
If they bet (which they will in a raised pot), the poison hand hits standard strategy which
folds equity < 0.30. The hand never reaches showdown. The data never gets poisoned. The flag
forces check-call to showdown regardless of opponent action, guaranteeing data pollution.

**Why Vector B is critical:** If poison only appears in passive/unraised pots, an opponent's
`analyze_logs.py` simply filters: `if no_raise_in_hand: continue`. This isolates ALL poison
into a clean dataset that can be discarded. By injecting poison into pots where WE raised
(e.g., we bluff-raised the flop, opponent called, we check-call the turn and river with
weak equity), the poison appears in the "raised pot" dataset — exactly the data opponents
use to calibrate our raise ranges. This is irreparably corrupting.

**Constraints:**
- Vector A: Not facing a raise (don't donate to aggression), equity < 0.35
- Vector B: `_poison_active` flag forces check-call to showdown. Only triggered when WE
  were the aggressor. equity < 0.30. Accepts the EV loss of calling down.
- Both vectors scale inversely with pot size
- Expected ~5-7 poison hands per 1000-hand match across all action lines

**EV cost:** ~6 hands × weighted avg ~8 chips = ~48 chips per match worst case (higher than
before because Vector B now actually reaches showdown via forced check-calls). Still small
compared to 100-500+ chips lost from being correctly read.

**Why this is unfilterable:** Poison appears in passive pots AND raised pots, across all
pot sizes. An opponent cannot filter by pot size, cannot filter by "raised vs unraised,"
and cannot filter by equity (they don't know ours). Their only option is to accept the
polluted data, which skews their Bayesian estimates of our call floors, raise ranges,
and showdown composition.

### Defense 4: Board-Texture Sizing (strategy.py)

**Threat:** If sizing correlates with hand strength (bigger = value, smaller = bluff),
k-means clustering on 50+ matches will separate our ranges perfectly.

**Fix:** Base bet sizing on BOARD TEXTURE, not hand strength. The same board produces
the same sizing whether we have the nuts or air:

```python
def _ace_wrap_connected(r1, r2):
    """Check if two ranks are connected, handling Ace wraps in 27-card deck."""
    diff = abs(r1 - r2)
    if diff <= 2:
        return True
    # Ace (rank 8) wraps: A-2 (diff=8), A-3 (diff=7) are connected
    # Also A-7, A-8, A-9 (Ace as 10 in 6-7-8-9-A straights)
    if r1 == 8 or r2 == 8:
        other = r2 if r1 == 8 else r1
        return other <= 1 or other >= 6  # 2,3 (low) or 7,8,9 (high)
    return False

def bet_size(pot, board, persona):
    """Size based on board wetness, not hand strength."""
    suits = [c // 9 for c in board]
    ranks = sorted([c % 9 for c in board])

    # Board wetness score: flush draws + straight draws + pairs
    flush_score = max(Counter(suits).values()) / len(board)  # 0.4-1.0
    # CRITICAL: Use ace-wrap-aware connectivity, not raw rank difference
    # Without this, A-2-3 board (ranks [0,1,8]) scores 0 connectivity
    # despite being a highly wet straight-draw board
    connect_score = sum(1 for i in range(len(ranks)-1)
                       if _ace_wrap_connected(ranks[i], ranks[i+1])) / max(1, len(ranks)-1)
    wetness = (flush_score + connect_score) / 2  # 0.0 = dry, 1.0 = wet

    # Wet boards → larger bets (more draws to charge / more bluff equity)
    # Dry boards → smaller bets (less to protect against)
    base = persona["bluff_size_center"] + wetness * 0.20
    base = max(0.40, min(0.95, base))

    # Jitter ±20% on top
    return int(pot * base * (1.0 + random.uniform(-0.20, 0.20)))
```

**Result:** Our bet size is a function of the board, not our cards. An opponent's scatter
plot of (bet_size vs hand_strength) shows zero correlation. Their clustering algorithm
finds nothing because there is nothing to find.

**EV impact:** Board-texture-based sizing is actually theoretically sound — GTO solvers
size larger on wet boards and smaller on dry boards. This isn't a sacrifice; it's better
poker AND better counter-intelligence.

### Defense Summary

| Defense | Where | EV Cost | Information Denied |
|---------|-------|---------|--------------------|
| Discard stochasticity | equity.py | <0.006/hand | Exact hole card deduction |
| Persona rotation | match_manager.py | ~0 | Cross-match threshold mapping |
| Showdown poisoning (2 vectors + flag) | strategy.py | ~48 chips/match | Range, call-floor, AND raise-range estimation |
| Board-texture sizing | strategy.py | ~0 (better poker) | Bet size ↔ hand strength correlation |

Total EV cost of all defenses: ~54 chips per 1000-hand match (~0.054 chips/hand).
Negligible compared to the 100-500+ chips lost from being correctly read by opponents.

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
- **time_per_hand** [float]: Adaptive. With vectorized exact equity taking ~200ms/hand worst case (500ms budget in Phase 1), time must be actively monitored. The rationality filter is the biggest cost; if running behind, switch to the two-tier pre-filter optimization.

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
     eq = equity.exact_equity(my_2, board, dead_cards, opp_discards, my_discards, opponent.opp_weights_fn)
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

For each 5-card hand, exhaustively enumerate ALL C(22,3) = 1,540 possible flops. For each flop,
evaluate all C(5,2) = 10 possible keeps using **flop-aware exact_equity(keep, flop, discard)** — NOT
hand_strength_2. Record the best keep's equity. Average over all flops.

**Why not hand_strength_2?** hand_strength_2 scores a 2-card hand's average equity across ALL possible
boards. Using it to choose the best keep after a specific flop is dealt commits the same "post-flop
blind" error fixed in the rationality filter — it ignores flush draws, straight draws, and all
board-specific equity. The generator must evaluate each keep against the actual flop.

80,730 entries. Total: 80,730 × 1,540 × 10 × ~10,920 = ~13.6T lookups. This is computationally
expensive (~hours to days) but runs OFFLINE once. Parallelizable across cores (each 5-card hand is
independent). Can generate incrementally and checkpoint.

**Truly exact — zero sampling noise, zero pre-flop-average approximation.**

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
- **hand_potential_5.npy generation time:** Flop-aware evaluation (10 keeps × exact_equity per keep) for each of 80,730 × 1,540 combos is computationally expensive (~hours to days). Mitigated by: (a) parallelizing across CPU cores (each 5-card hand is independent), (b) checkpointing progress, (c) can ship v1 with hand_strength_2-based approximation and upgrade to exact table in v2.
