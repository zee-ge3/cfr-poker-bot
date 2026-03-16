STRATEGIC ANALYSIS FROM THE ANALYST SESSION — READ EVERYTHING BEFORE MAKING CHANGES
===================================================================================

Analyzed all 7 AVBv10 tournament losses (matches 4712, 4940, 5013, 5038, 5042, 5090, 5130)
plus the full tournament dataset (334 PlayerAgent matches, 34 Claude2Agent matches). AVBv10 is
currently 19/26 (73.1% WR) in live tournament. Here are the findings:

## CRITICAL FLAW 1: POST-FLOP FOLD-TO-RAISE RATE (75-83%)

This is the number 1 cause of every single loss. The FACING A BET section (lines 749-805) is too tight:

- aggr_call_floor defaults to 0.50, drops to 0.40 floor only at extreme aggression (rr>0.45).
  Most post-flop hands have MC equity 0.35-0.50, so they ALL fold.
- rule4_thr is max(0.38, 0.47 - ...) — floors at 0.38, defaults 0.47
- rule5 only applies at pot_odds <= 0.25 — a min-raise of 5 into 15 is already 0.25

RESULT across all 7 losses:
  Match 4712 vs Qwerty:         fold-to-raise 74.8%, opp raised 1300x, geoz raised 601x
  Match 4940 vs Me & Claude <3: fold-to-raise 74.5%, opp raised 1079x, geoz raised 450x
  Match 5013 vs GradientAscent: fold-to-raise 73.4% (82.1% on FLOP specifically)
  Match 5038 vs AlbertLuoLovers: fold-to-raise 76.7% (82.7% on flop)
  Match 5042 vs sheep army:     fold-to-raise 53.6%
  Match 5090 vs sheep army:     fold-to-raise 64.3%
  Match 5130 vs sheep army:     fold-to-raise 61.7%

Opponents exploit this by raising every flop for 5 chips. Win rate: 75-83%.
Math: bet 5 to win 6-8, succeed 80% of the time = +3.8 chips/attempt x 400 attempts = +1500 chips.

FIX NEEDED:
- aggr_call_floor should drop to approximately 0.36-0.38 when rr > 0.45 (not 0.40)
- rule4_thr floor should be approximately 0.33 against aggressive opponents (not 0.38)
- rule5 pot_odds limit should extend to 0.30+ (not 0.25)
- These should scale smoothly with opponent raise frequency, not use hard cutoffs

## CRITICAL FLAW 2: _opp_raises_often() DOES NOT AFFECT POST-FLOP CALLING

The function (line 144) requires raise rate > 0.48 to trigger. Even when it does, it ONLY:
- Widens preflop call range (line 542-543)
- Suppresses thin probes (line 621/678)

It does NOT lower post-flop calling thresholds. The bot recognizes hyper-aggression but
DOES NOT CHANGE ITS FOLD BEHAVIOR. It just stops probing (reduces its own offense) while
still folding to opponent offense. This is exactly backwards.

FIX: When raises_often is True, ALL facing-bet thresholds should drop significantly.

## CRITICAL FLAW 3: _opp_bluff_rate() IS NEVER USED IN _decide()

The function exists (lines 130-142) and returns bluff rate estimates, but it is NEVER
referenced anywhere in _decide(). It caps at 0.15 even for opponents raising 70%+ of hands.
In reality, if someone raises 70% of flops, at least 40-50% are bluffs.

FIX: Use bluff_rate to adjust calling thresholds. If bluff_rate > 0.20, widen call range.

## CRITICAL FLAW 4: NO BET SIZE TRACKING

_opp_raise_count increments identically for a 5-chip min-raise and a 95-chip pot shove.
A min-raise carries completely different information than a pot-size raise. The bot makes
identical decisions against both.

FIX: Track opponent bet sizes. When they min-raise (less than 0.3x pot), treat as probe/bluff
and call wider. When they pot-size raise (greater than 0.8x pot), treat as value and fold tighter.

## CRITICAL FLAW 5: NO STREET-TO-STREET LINE TRACKING

The bot has no concept of opponent action sequences within a hand:
- "check flop, raise turn" = classic slowplay/trap — should be respected more
- "raise flop, raise turn" = continued aggression — more likely value OR persistent bluff
- "check flop, check turn, raise river" = caught card or bluff

Each street is evaluated independently. _checked_this_street and _bet_this_street track
OUR actions, not the opponent actions. There is no _opp_action_history per hand.

FIX: Track the opponent action sequence per hand. Check-raise lines deserve more respect
(fold more). Bet-bet-bet lines from hyper-aggressive opponents deserve less respect (call more).

## FLAW 6: HARD THRESHOLD CLIFFS INSTEAD OF PROBABILISTIC GRAY ZONES

The calling thresholds are hard cutoffs:
- equity 0.499 vs aggr_call_floor 0.50 = ALWAYS fold
- equity 0.501 = ALWAYS call
- equity 0.629 = check, equity 0.631 = bet 0.80x pot

This creates exploitable patterns. An opponent estimating your equity range can predict
exactly when you will fold. All thresholds should have probabilistic transition zones.

Instead of: if equity >= 0.50: call() else: fold()
Use something like: call_prob = sigmoid((equity - threshold) / temperature)
     if random.random() < call_prob: call() else: fold()

This makes the bot much harder to exploit while maintaining the same average behavior.
Use wider transition zones (higher temperature) for less critical decisions and narrower
zones for critical ones. IMPORTANT: Do not use uniform noise — sigmoid or logistic gives
smooth probability curves that are mathematically sound.

## FLAW 7: NO EARLY GAME AGGRESSION / PHASE-BASED STRATEGY

This is the biggest strategic insight. The match structure creates an asymmetry:
- Stack resets to 100 every hand, so you can NEVER be felted
- 1000 hands at 1.5 chips/hand blind bleed = plenty of runway
- Losing 150 chips on hand 10 costs nothing (990 hands to recover)
- Winning 150 chips on hand 10 creates a chokehold (opponent cannot recover if you protect)
- If opponent ALSO has autofold/protection, whoever gets the early lead wins the match

The bot currently plays identically on hand 1 and hand 900 (plus or minus 0.04 pressure adjustment).
It should play FUNDAMENTALLY differently:

EARLY GAME (hands approximately 0-150):
- Wide pre-flop opens (lower pf_open_thr and pf_call_thr)
- Take coin-flip equity spots (call with equity 0.45+ even at higher pot odds)
- Aggressive semi-bluffs on flop/turn
- Larger value bets to build bigger pots
- Accept higher variance — every big pot won early is worth more than the same pot late

MID GAME (hands approximately 150-500):
- If ahead: progressively tighten. Reduce bluffs, reduce thin value bets.
- If behind: stay aggressive, similar to early game profile.

LATE GAME (hands approximately 500+):
- If ahead: protection mode, ultra-tight, autofold when lead is safe
- If behind: maximum desperation — widest calls, most bluffs, biggest bets

CRITICAL: Do NOT use hard cutoffs between phases. The transitions must be PROGRESSIVE and
RANDOMIZED so opponents cannot detect the phase transition. Use smooth interpolation:

```python
hands_played = self._hands_played
# Smooth aggression curve: high early, decays with lead
early_factor = max(0.0, 1.0 - hands_played / 300)  # 1.0 at hand 0, 0.0 at hand 300
# Add randomness: plus or minus 20% jitter on the factor
early_factor *= (1.0 + random.uniform(-0.20, 0.20))
# Combine with pressure: if behind, stay aggressive regardless of hand count
aggression = max(early_factor, max(0.0, pressure) * 0.8)
```

Then USE aggression to continuously scale all thresholds:
```python
pf_call_thr -= aggression * 0.06    # call wider early
aggr_call_floor -= aggression * 0.06 # call raises wider early
bluff_freq += aggression * 0.15      # bluff more early
value_bet_size *= (1 + aggression * 0.2) # bet bigger early
```

This creates a smooth gradient from "maniac" (hand 1) to "GTO" (hand 300+) to "rock"
(hand 500+ with lead), with enough randomness that opponents cannot exploit the transition.

## WHAT WAS ADDED THAT IS GOOD (keep these)

1. Showdown score tracking (_opp_showdown_scores, _opp_showdown_tight) — good passive info
2. Suboptimality bonus (_opp_subopt_bonus_sum) — good equity correction
3. Dynamic MC sample scaling — better time management

BUT _opp_showdown_tight() makes the fold problem WORSE against aggressive opponents:
it raises reraise_thr by +0.04 when opponent shows strong hands. Against someone who
raises 70% of hands, the showdowns will look strong (because you only reach showdown
when THEY have value). This biases the sample and tightens you further. Consider: only
use showdown tightness data from hands where the opponent did NOT raise (checked/called
their way to showdown). That is their true value range.

## WHAT STILL IS NOT TRACKED (from last session recommendations)

1. Opponent discard patterns across hands — you see 3 of their 5 cards every hand.
   If they discard three different suits with no pairs, they likely kept a pair or
   two suited cards. If they discard a pair, they must have something better.
   Track: how often do they keep suited? pairs? high cards? This narrows range before showdown.

2. Opponent bet sizing — not just whether they raised, but HOW MUCH. Min-raise = probe.
   Pot-raise = value. Track average raise size per street.

3. Opponent action sequences per hand — check-raise vs bet-bet vs check-check-bet patterns.

## COMPARISON: PlayerAgent (AEAv18) vs Claude2Agent (AVBv10)

PlayerAgent has 80.5% WR (269/334 in logs) vs AVBv10 73.1% (19/26 dashboard).
Key behavioral differences that make PA more successful:

- PA pre-flop fold rate: 29.7% vs C2: 10.0% — PA is much more selective pre-flop
- PA raises per match: 569 vs C2: 929 — PA raises less but more meaningfully
- PA avg showdown result: +10.0 chips vs C2: +5.6 — PA wins more when it plays
- PA avg pot at showdown: 58.4 vs C2: 97.1 — C2 builds pots 66% bigger but wins less

C2 enters too many pots with marginal holdings and inflates pots it then folds or loses.
PA plays fewer hands but wins bigger when it plays.

## SUMMARY OF PRIORITY CHANGES (in order)

1. LOWER POST-FLOP CALL THRESHOLDS vs frequent raisers (biggest WR impact)
2. ADD EARLY GAME AGGRESSION with smooth progressive transitions (strategic edge)
3. ADD PROBABILISTIC GRAY ZONES on all thresholds (anti-exploit)
4. MAKE raises_often AFFECT post-flop calling (currently does nothing useful)
5. TRACK AND USE bet sizing (min-raise vs pot-raise should trigger different responses)
6. TRACK opponent action sequences per hand (line-based reads)
7. TRACK opponent discard patterns across hands (range narrowing)
8. FIX showdown_tight bias (only use non-raise showdowns for range estimation)

The user wants to submit this bot. Focus on items 1-4 first as they directly address
the identified loss patterns. Items 5-8 are important but secondary.
