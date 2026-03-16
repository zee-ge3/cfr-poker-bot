ANALYST FINDINGS FOR PLAYERAGENT v18r — TOURNAMENT LOSS ANALYSIS
================================================================

PA v18r is posting 4/6 (67% WR). Two new losses analyzed below.
The v18r rewrite fixed the catastrophic 0.62 flop threshold from match 4683.
Post-flop fold discipline is now good (avg fold equity 0.17, was 0.48).
But two NEW failure modes are exposed that need fixing.

## LOSS 1: Match 5944 vs "Alan Keating" — LOST -753

Chip trajectory: peaked +83 at hand 45, then bled out. Never recovered.
Two major collapses: H500-600 (-325), H900-1000 (-185).

### Numbers:
- PA PF fold rate: 49.8% (498 folds out of ~1000 hands)
- Opponent PF fold rate: 0.8% (7 folds total)
- PA post-flop fold-to-raise: 37.3% (reasonable)
- Opponent post-flop fold-to-raise: 57.3%
- PA total raises: 348, Opponent total raises: 753
- PA PF raises: only 20 out of 1000+ hands

### Root causes:
1. OPPONENT MISCLASSIFICATION: Bot labeled opponent "tag" 98.6% of the time.
   But Alan Keating played 99.2% of hands and raised PF ~50%. This is LAG/maniac.
   Wrong label → wrong thresholds → wrong decisions for the entire match.

2. EQUITY MODEL OVERCONFIDENCE: In ALL 5 biggest losses (each -100 chips), PA held
   equity 0.59-0.98 at final decision but lost to flushes/trips/full houses:
   - H67: 9s9d, eq=0.68 → lost to 7d2d flush
   - H73: AdAs, eq=0.70 → lost to 6h4h flush (three hearts on board!)
   - H186: 5h5s, eq=0.94 → lost to 8s7h trips (trip 5s vs trip 8s, cooler)
   - H218: Ah9h, eq=0.87 → lost to 4sAd trips (equity was WRONG)
   - H434: AdAh, eq=0.81 → lost to 8d7h full house on paired board

   The MC equity systematically overestimates one-pair and two-pair hands on wet
   boards (3+ of one suit, paired boards). PA's importance sampling helps but
   doesn't account for board texture danger.

3. 49.8% PF FOLD RATE: Against an opponent raising 50% with 0.8% fold rate,
   each PF fold costs ~1.84 chips. 498 folds × 1.84 = ~916 chips leaked from
   blind steals. This is the #1 chip drain.

4. ONLY 20 PF RAISES: PA opened pre-flop only 20 times. Against an opponent
   folding 0.8% PF, this is correct (no fold equity). But PA should be CALLING
   more (not folding 50%) and raising MORE POST-FLOP where the opponent folds 57%.

### What needs to change for this matchup:
- Fix opponent type classification: an opponent playing 99% of hands is NOT "tag"
- Lower PF call threshold more aggressively against LAG opponents (currently -0.13
  max LAG adjustment, but 49.8% fold rate means it's not enough)
- Discount equity on wet boards: 3+ of one suit → pair equity drops 15-20%

## LOSS 2: Match 5913 vs "Ctrl+Alt+Defeat" — LOST -1422

This is the SAME opponent that beat PA -786 in match 5801. Same strategy, worse result.

### Numbers:
- PA PF fold rate: 40.5% (450 folds)
- Opponent PF raise rate: 91% of hands (583 raises, avg 11.2 chips)
- Opponent post-flop fold-to-raise: 82.2% ← THIS IS THE KEY NUMBER
- PA total raises: 266, Opponent total raises: 1013 (4x more)
- PA post-flop fold equity at fold: avg 0.174 (disciplined — NOT folding winners)
- PA PF fold equity at fold: avg 0.312

### Root cause — THE CRITICAL ONE:

THE OPPONENT FOLDS 82% TO POST-FLOP RAISES BUT PA ONLY RAISED 266 TIMES.

This is a massive exploitation gap. When opponent folds 82% to raises:
- A 5-chip raise into a 10-chip pot succeeds 82% of the time
- EV = 0.82 * 10 - 0.18 * 5 = 8.2 - 0.9 = +7.3 chips per raise (pure bluff!)
- Even raising with 0% equity is massively +EV
- PA should be raising EVERY SINGLE STREET against this opponent

PA's `folds_often` detection exists and works. It sees the opponent folds frequently.
But it only uses this information to:
1. Adjust call thresholds (irrelevant when opponent rarely raises back)
2. Slightly increase bluff frequency (not enough)

It does NOT use it to fundamentally change from "call/fold reactive" to
"raise constantly to exploit fold equity."

### Specific code change needed:

In `_act_betting()`, when `folds_often` is True AND the opponent folds to post-flop
raises at a very high rate (say >65%), PA should:

1. PROBE (raise) on EVERY street where it has the initiative and hasn't been raised
   - Current: only probes when equity > probe_threshold (~0.50-0.55)
   - Needed: probe when equity > 0.20 if fold_rate > 0.65 (pure bluff profitable)
   - Even better: probe with probability = fold_rate when equity > 0.15

2. RAISE (re-raise) facing bets more often when their fold-to-reraise is high
   - Current: only re-raises at reraise_thr (~0.62+)
   - Needed: re-raise at 0.35+ equity if their fold-to-reraise > 0.50

3. SIZE raises larger when fold rate is high
   - Current: 0.75x pot standard
   - Needed: 0.50x pot when exploiting folds (cheaper bluff with same fold rate)

### The math:

Against Ctrl+Alt+Defeat (82% fold rate):
- PA raised 195 times post-flop across 3 streets
- PA could have raised approximately 600+ times (every street in every hand)
- At 82% success rate: 400 extra successful bluffs × avg 8 chips = +3200 chips
- Minus 110 failed bluffs × avg 5 chips = -550 chips
- Net improvement: approximately +2650 chips
- This ALONE would have flipped the -1422 loss into a +1200 win

## LOSS PATTERN: Match 4683 vs WW — CONTEXT

PA previously lost -3388 to WW (match 4683) with 63% flop fold rate and average
fold equity of 0.482. The v18r rewrite FIXED this — fold equity at fold is now
0.174 (disciplined). But the match 5913 loss shows the NEW problem: PA stopped
folding winners, but didn't start RAISING to exploit the fold equity gap.

## SUMMARY OF ALL REMAINING CHANGES

Priority order:

1. RAISE FREQUENCY vs FOLDERS (highest impact — match 5913 alone worth +2650 chips)
   - When fold_rate > 0.60: probe with equity > 0.20 (not 0.50)
   - When fold_rate > 0.70: probe with ANY hand on unchecked streets
   - Scale raise frequency linearly with opponent fold rate
   - Use smaller sizing (0.50x pot) for bluff-raises to minimize risk

2. OPPONENT TYPE CLASSIFICATION FIX (match 5944)
   - An opponent playing 99% of hands and raising 50% PF is LAG, not TAG
   - The classification thresholds are wrong. Check the VPIP/RR boundary
     between tag and lag. Currently seems to require high RR but doesn't
     account for high VPIP (voluntarily putting money in pot rate).

3. WET BOARD EQUITY DISCOUNT (match 5944 — 5 coolers on flush/paired boards)
   - When board has 3+ of one suit: discount one-pair equity by 15-20%
   - When board is paired: discount trip equity by 10% (full house possible)
   - This is already partially handled by _opp_keep_weight() importance sampling
     but the MC still overestimates one-pair hands systematically

4. PRE-FLOP CALL THRESHOLD vs HYPER-AGGRESSIVE (both matches)
   - PF fold rate 40-50% vs opponents raising 50-91% is still too high
   - Against a 91% raiser, their average hand is trash — calling with 0.30
     equity is +EV because their range is so wide
   - The LAG PF adjustment (-0.13 max) needs to scale more aggressively:
     at PF raise rate > 0.70, call threshold should drop to 0.15-0.20

## WHAT v18r GOT RIGHT (keep these)

1. Post-flop fold discipline: avg fold equity 0.17 — no longer folding winners
2. Raise war cap: prevents escalation spirals
3. Importance sampling: _opp_keep_weight() correctly models opponent preferences
4. Showdown calibration: adjusts call thresholds based on what opponent shows
5. Bet sizing tell: correlates bet size with hand strength
6. Time management: using only 10-15% of budget, plenty of room

## CROSS-BOT COMPARISON

Both PA and C2 face the same opponent pool. Key differences:

Against "Ctrl+Alt+Defeat":
- PA lost -786 (5801) and -1422 (5913) — opponent folds 82% post-flop but PA
  doesn't exploit it
- C2 hasn't faced this opponent yet

Against "WW" (hyper-aggressive):
- PA lost -3388 (4683) — catastrophic fold hemorrhage (v18 bug, now fixed)
- C2 WON +73 (5606) — C2's lower call thresholds let it call down WW's bluffs

Against "AlbertLuoLovers":
- PA lost -9 (4786) — nearly recovered from opponent autofold
- C2 lost -5 (5633) — nearly recovered from opponent autofold
- Both bots lose marginally to this opponent's strategic lock-in timing

The gap between PA and C2 is NOT call thresholds anymore (v18r fixed that).
The gap is RAISE FREQUENCY. PA needs to raise more against folders, period.
