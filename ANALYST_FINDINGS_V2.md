ANALYST FINDINGS V2 — REMAINING FIXES NEEDED BEFORE SUBMISSION
================================================================

Good work on the first round of changes. The post-flop call threshold fixes, early game
aggression curve, sigmoid gray zones, and showdown bias fix all look correct.

This document covers: 1 bug to fix, and 4 remaining features that need implementation.

## BUG: _opp_raised_this_hand NOT RESET ON NON-SHOWDOWN HANDS

Line 1055 resets `self._opp_raised_this_hand = False` only inside the
`if "player_0_cards" in info:` block (showdown). If a hand ends in a fold (no showdown),
the flag stays True and contaminates the NEXT hand's showdown classification.

Example: Hand 50 — opponent raises, we fold (no showdown, flag stays True).
Hand 51 — opponent plays passively, reaches showdown. But the stale flag from hand 50
marks it as a raising hand, so it gets EXCLUDED from _opp_passive_sd_scores.
This corrupts the showdown tightness data over time.

FIX: Also reset the flag in the terminated block at the top of observe().
Add this at the end of the `if terminated:` block in observe() (around line 997):

```python
        if terminated:
            self._hands_played += 1
            self._cumulative_reward += reward
            self._opp_raised_this_hand = False   # <-- ADD THIS LINE
```

## MISSING FEATURE 1: OPPONENT BET SIZE TRACKING

Currently _opp_raise_count increments identically for a 5-chip min-raise and a 95-chip
pot shove. The bot makes identical calling decisions against both. This is a massive
information leak.

WHAT TO TRACK: In the opponent model section, add:

```python
# In __init__:
self._opp_raise_sizes = []  # list of (raise_amount, pot_size) tuples
```

Record every opponent raise with its size relative to pot. In the action counting section
of act() where you detect "RAISE" in opp_last, also record the size:

```python
if "RAISE" in opp_last:
    self._opp_raise_count += 1
    self._opp_raised_this_hand = True
    # Track bet sizing
    opp_raise_size = obs["opp_bet"] - obs["my_bet"]
    if pot > 0 and opp_raise_size > 0:
        self._opp_raise_sizes.append((opp_raise_size, pot))
    if len(self._opp_raise_sizes) > 200:
        self._opp_raise_sizes = self._opp_raise_sizes[-200:]
```

Then add a method to compute average raise sizing:

```python
def _opp_avg_raise_frac(self) -> float:
    """Average opponent raise as fraction of pot. <0.35 = min-raiser/probe, >0.70 = value-heavy."""
    if len(self._opp_raise_sizes) < 5:
        return 0.50  # neutral prior
    return sum(sz / max(1, p) for sz, p in self._opp_raise_sizes) / len(self._opp_raise_sizes)
```

HOW TO USE IN _decide(): When facing a bet, check the CURRENT raise size vs their average:
- Current raise is small (< 0.3x pot) AND they typically min-raise: this is a probe/bluff,
  call wider (lower aggr_call_floor by 0.04)
- Current raise is large (> 0.8x pot) AND they typically make big raises: this is value,
  fold tighter (raise aggr_call_floor by 0.03)
- Current raise is large but they typically min-raise: this is polarized (either monster
  or big bluff), use standard thresholds

Add to _decide() before the FACING A BET section:

```python
# Bet size read: current raise vs opponent average
if facing and cost > 0:
    current_raise_frac = cost / max(1, pot - cost)  # raise relative to pot before raise
    avg_frac = self._opp_avg_raise_frac()
    if current_raise_frac < 0.30 and avg_frac < 0.40:
        # Habitual min-raiser making another min-raise = likely probe/bluff
        aggr_call_floor -= 0.04
    elif current_raise_frac > 0.80:
        # Big raise = more likely value
        aggr_call_floor += 0.03
```

## MISSING FEATURE 2: OPPONENT ACTION LINE TRACKING (per-hand sequences)

The bot evaluates each street independently with no memory of what the opponent did on
earlier streets within the same hand. This misses critical patterns:

- "check flop, raise turn" = classic slowplay/trap line, should be respected MORE
- "raise flop, raise turn, raise river" = either monster or persistent bluff
- "check flop, check turn, raise river" = either caught a card or bluffing the scare card

WHAT TO TRACK:

```python
# In __init__:
self._opp_hand_actions = []  # per-street actions this hand: ['CHECK', 'RAISE', ...]
```

Reset at hand start (in the terminated block or when street resets to 0).
Append opponent actions as they come in.

Then add a method:

```python
def _opp_line_is_trap(self) -> bool:
    """Detect check-raise line: opp checked earlier street, now raising = trap."""
    return 'CHECK' in self._opp_hand_actions[:-1] and self._opp_hand_actions[-1] == 'RAISE'

def _opp_line_is_barrel(self) -> bool:
    """Detect multi-barrel: opp raised on 2+ streets = persistent aggression."""
    return sum(1 for a in self._opp_hand_actions if a == 'RAISE') >= 2
```

HOW TO USE: In the facing-bet section:
- If _opp_line_is_trap(): raise aggr_call_floor by +0.04 (respect the trap)
- If _opp_line_is_barrel() and raises_often: lower aggr_call_floor by -0.03
  (persistent aggression from a hyper-raiser = more likely bluff continuation)

## MISSING FEATURE 3: OPPONENT DISCARD PATTERN TRACKING

Every hand, you see 3 of the opponent's 5 dealt cards (their discards). This is massive
free information that is currently thrown away after a single hand's MC exclusion.

WHAT TO TRACK:

```python
# In __init__:
self._opp_discard_data = {
    'kept_suited_count': 0,   # times discards suggest they kept 2 suited cards
    'kept_pair_count': 0,     # times discards suggest they kept a pocket pair
    'discarded_pair_count': 0, # times they threw away a pair (have something better)
    'total_observed': 0,
}
```

Analyze discards each hand during the DISCARD phase or when opp_discarded_cards are first seen:

```python
def _analyze_opp_discards(self, opp_discards):
    """Deduce what opponent likely kept from their 3 discarded cards."""
    if len(opp_discards) != 3 or any(c == -1 for c in opp_discards):
        return
    self._opp_discard_data['total_observed'] += 1

    ranks = [c % NUM_RANKS for c in opp_discards]
    suits = [c // NUM_RANKS for c in opp_discards]

    # Did they discard a pair? If so, they kept something better
    from collections import Counter
    rank_counts = Counter(ranks)
    if max(rank_counts.values()) >= 2:
        self._opp_discard_data['discarded_pair_count'] += 1

    # Did they keep suited cards? If all 3 discards have different suits,
    # the 2 kept cards must share a suit with at most 1 discard.
    # If all 3 discards are from 2 suits, the kept cards likely have the 3rd suit.
    suit_counts = Counter(suits)
    if len(suit_counts) == 3:
        # All 3 different suits discarded - kept cards likely suited (same suit as each other)
        self._opp_discard_data['kept_suited_count'] += 1

    # Did they keep a pair? If no rank appears twice in discards, AND they didn't
    # keep suited (above), they likely have a pocket pair.
    if max(rank_counts.values()) == 1 and len(suit_counts) < 3:
        self._opp_discard_data['kept_pair_count'] += 1
```

Call this from act() right after parsing opp_discards, once per hand (on the discard street
or the first post-flop street).

HOW TO USE:

```python
def _opp_is_flush_chaser(self) -> bool:
    """True if opponent frequently keeps suited cards (flush-draw bias)."""
    total = self._opp_discard_data['total_observed']
    if total < 15:
        return False
    return self._opp_discard_data['kept_suited_count'] / total > 0.45

def _opp_is_pair_keeper(self) -> bool:
    """True if opponent strongly prefers keeping pocket pairs."""
    total = self._opp_discard_data['total_observed']
    if total < 15:
        return False
    return self._opp_discard_data['kept_pair_count'] / total > 0.40
```

Then in _decide():
- If _opp_is_flush_chaser() and board has 3+ of one suit: their raise is more credible
  on flush boards (raise aggr_call_floor by +0.03 on flush-heavy boards)
- If _opp_is_flush_chaser() and board has NO flush potential: their raise is more likely
  a bluff (lower aggr_call_floor by -0.02)
- If _opp_is_pair_keeper() and board is paired: they likely have trips/FH, respect
  their raises more (raise aggr_call_floor by +0.03)

## MISSING FEATURE 4: OPPONENT SHOWDOWN HAND TYPE TRACKING

Currently you record the treys score but NOT what type of hand the opponent showed down with.
Knowing "this opponent showed Flush 6 out of 10 times" vs "this opponent showed Pair 8 out
of 10 times" is extremely valuable for calibrating how to play against them.

WHAT TO TRACK:

```python
# In __init__:
self._opp_sd_hand_types = {}  # {'flush': 5, 'pair': 8, 'trips': 2, ...}
```

In observe() after computing the showdown score, classify and record:

```python
# After: score = self.evaluator.evaluate(opp_kept_t, board_t)
if score <= 10:      hand_type = 'straight_flush'
elif score <= 322:   hand_type = 'full_house'
elif score <= 1599:  hand_type = 'flush'
elif score <= 1609:  hand_type = 'straight'
elif score <= 2467:  hand_type = 'trips'
elif score <= 3325:  hand_type = 'two_pair'
elif score <= 6185:  hand_type = 'pair'
else:                hand_type = 'high_card'
self._opp_sd_hand_types[hand_type] = self._opp_sd_hand_types.get(hand_type, 0) + 1
```

HOW TO USE:

```python
def _opp_sd_flush_rate(self) -> float:
    """Fraction of showdowns where opponent had flush or better."""
    total = sum(self._opp_sd_hand_types.values())
    if total < 8:
        return 0.20  # neutral
    flush_plus = (self._opp_sd_hand_types.get('flush', 0) +
                  self._opp_sd_hand_types.get('full_house', 0) +
                  self._opp_sd_hand_types.get('straight_flush', 0))
    return flush_plus / total
```

If opponent shows down with flushes over 35% of the time AND the current board is flush-heavy:
respect their raises more. If they rarely show flushes: their raises on flush boards are
more likely bluffs.

## SUMMARY OF ALL REMAINING CHANGES

1. BUG FIX: Reset _opp_raised_this_hand in observe() terminated block (1 line)
2. Bet size tracking + use in calling decisions (new field, new method, 5-10 lines in _decide)
3. Action line tracking per hand (new field, reset logic, 2 methods, 3-5 lines in _decide)
4. Discard pattern tracking (new field, analysis method, 2 classification methods, 3-5 lines in _decide)
5. Showdown hand type tracking (new field, classification in observe, rate methods, 2-3 lines in _decide)

All of these feed into aggr_call_floor adjustments — they make the bot's calling decisions
context-aware instead of one-size-fits-all. The information is already available in the
observation data, it just needs to be recorded and used.

Priority order: Bug fix > Bet sizing > Action lines > Discard patterns > Hand types
(bet sizing has the most immediate impact on the fold-to-raise vulnerability)
