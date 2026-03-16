# Info-Extraction Pipeline Design
**Date:** 2026-03-15
**Context:** CMU AI Poker Tournament — 6 days to submission deadline
**Goal:** Maximize intelligence gain on opponent strategies via an overnight data-collection run, fully automated log pipeline, and structured opponent profiling.

---

## Overview

Three components built together as a single overnight intelligence pipeline:

1. **Spy Bot** (`agents/spy_agent.py` / `submission/spy_player.py`) — submitted temporarily to gather opponent data
2. **Playwright Automation Daemon** (`auto_playwright.py`) — handles bot submission, session refresh, and log fetching unattended overnight
3. **Opponent Profiler** (`opponent_profiler.py`) — reads overnight logs, cross-references leaderboard, outputs per-opponent exploit profiles

**Workflow:**
- Before bed: run `--setup`, swap in spy bot, start daemon
- Overnight: daemon fetches logs every 15 min, leaderboard snapshots tagged to each match
- Morning: swap back main bot, run profiler, read exploit report

---

## Component 1: Spy Bot

### Purpose
Not to win chips — to expose opponent strategy patterns across three dimensions:
- **A) Fold-to-raise rates** (by street, position, raise size)
- **B) Bet sizing tells** (what they bet with strong vs. weak hands)
- **C) Calling/folding thresholds** (showdown hand strength distributions)

### Position-Based Mode Selection
Each hand, the bot selects a primary mode based on position:
- **IP (button, acts last post-flop):** Raise mode — probe every street to measure fold-to-raise
- **OOP (big blind, acts first post-flop):** Call mode — call all bets to showdown to collect hand strength + sizing data

Position detection: compare `obs["my_bet"]` vs `obs["opp_bet"]` at street 0.

### Disguise Layer
Opponents may model the spy bot and adapt, contaminating data. To prevent this:
- **Mode randomization:** 20% of hands flip the mode (IP hand plays call mode, OOP plays raise mode)
- **Raise sizing noise:** randomize between 0.3x–1.2x pot (not always 0.5x)
- **Occasional folds:** ~10% of strong-hand situations fold to appear non-robotic
- **Occasional limps:** ~15% of IP pre-flop hands limp instead of raising

No opponent should be able to reliably detect the IP/OOP pattern within a 1000-hand match.

### Position-Stratified Logging
Every action tagged with four fields written to agent log:
- `our_pos`: IP or OOP
- `opp_pos`: IP or OOP (derived — opposite of ours)
- `street`: 0–3 (pre-flop through river)
- `hand_num`: hand number within match

This enables the profiler to ask position-stratified questions (e.g., "how often does opponent X fold to a raise when *they* are OOP on the turn?").

### Adaptation Detection
Track opponent fold-to-raise rate in hands 1–500 vs. 501–1000. If the delta exceeds 15 percentage points, flag the match as "opponent adapted." Adapted matches are down-weighted in profiler aggregation.

---

## Component 2: Playwright Automation Daemon

### File: `auto_playwright.py`

### Setup (interactive, run once before bed)
```
python auto_playwright.py --setup
```
- Launches visible Chromium
- Navigates to `aipoker.cmudsc.com`
- Waits for manual login
- Saves full browser storage state (cookies + localStorage including Clerk `__client` long-lived token) to `.browser_state.json`
- Validates submission page UI selectors (file input, submit button) and saves selector config

### Bot Submission
```
python auto_playwright.py --submit <path_to_zip_or_py>
```
- Loads `.browser_state.json` headlessly
- Builds zip if a `.py` file is passed (using `create_release.sh` logic)
- Navigates to submission page, uploads file, confirms success
- Used to swap spy bot in before bed and main bot back in the morning

### Daemon Mode
```
python auto_playwright.py --daemon --interval 15
```
Every 15 minutes:
1. Load saved browser state headlessly
2. Navigate to dashboard — Clerk client token auto-refreshes `__session` JWT
3. Extract fresh `__session` cookie from browser context
4. Call `auto_fetch_logs.run_fetch(cookie)` directly (no subprocess)
5. Fetch current leaderboard, snapshot ranks to `overnight_log.jsonl`
6. Append new match IDs with opponent name + their rank at time of match

If auth fails (session revoked): write `.auth_expired` flag file, emit terminal bell, continue polling without fetching.

### Leaderboard Tagging
Each fetch cycle writes a JSONL line:
```json
{"ts": "2026-03-16T02:15:00", "match_id": 6123, "opponent": "WW", "opp_rank": 1, "opp_elo": 1842, "result": "LOST", "net": -312}
```
This lets the profiler separate analysis by opponent tier (top-20 vs. bottom-20).

---

## Component 3: Opponent Profiler

### File: `opponent_profiler.py`

### Input
- All CSVs in `tournament_logs/` tagged as spy-bot matches (by bot name in filename)
- `overnight_log.jsonl` for leaderboard rank snapshots
- Agent logs for position-tagged action data

### Metrics Computed Per Opponent
| Metric | Description |
|--------|-------------|
| `pf_fold_rate` | Pre-flop fold rate |
| `vpip` | Voluntarily put money in pot rate |
| `ftr_oop_{street}` | Fold-to-raise when opponent is OOP, by street |
| `ftr_ip_{street}` | Fold-to-raise when opponent is IP, by street |
| `avg_bet_frac_{street}` | Average bet as fraction of pot, by street |
| `sd_hand_dist` | Showdown hand type distribution (pair/two-pair/trips/flush/etc.) |
| `type` | Classified opponent type: TAG / LAG / calling-station / maniac |
| `adapts` | Whether fold-to-raise rate dropped >15pp in second half |

### Output: Ranked Table
Sorted by current leaderboard rank, minimum 30 hands of data:
```
Rank  Opponent            PF-fold  FTR-OOP  FTR-IP  AvgBet  Type     Adapts?
#1    WW                   12%      82%      45%     1.1x    maniac   no
#5    GradientAscent       31%      58%      39%     0.7x    lag      yes
#12   sheep army           22%      64%      51%     0.5x    lag      no
```

### Auto-Generated Exploit Notes
For each top-20 opponent with ≥30 hands:
```
WW (#1): folds 82% OOP post-flop → raise every street when they're OOP, small sizing (0.5x) for max EV
GradientAscent (#5): adapts ~hand 200, large bets = value → bluff-raise heavy in first 200 hands only
sheep army (#12): folds 64% OOP, 51% IP → consistent folder, probe all streets
```

These notes are directly usable for tuning the main bots or writing targeted counter-strategies before the submission deadline.

---

## New Files

| File | Purpose |
|------|---------|
| `agents/spy_agent.py` | Info-extraction bot (local testing) |
| `submission/spy_player.py` | Tournament submission version |
| `auto_playwright.py` | Playwright daemon: setup, submit, fetch |
| `opponent_profiler.py` | Overnight log analysis + exploit report |
| `.browser_state.json` | Saved Playwright session (gitignored) |
| `overnight_log.jsonl` | Per-fetch leaderboard + match snapshots |

## Files Modified

| File | Change |
|------|--------|
| `auto_fetch_logs.py` | Expose `run_fetch()` as callable (already partially done) |
| `.gitignore` | Add `.browser_state.json`, `.session_cookie`, `.auth_expired` |

---

## Constraints & Risks

- **Spy bot ELO cost:** Running overnight will likely lose chips (passive/random play). This is acceptable — ELO volatility is high and the intelligence gain outweighs short-term rank drop.
- **Playwright session expiry:** If Clerk revokes the session mid-night (e.g., after ~8h), the daemon stops fetching but does not crash. Data already collected is preserved.
- **Submission page selectors:** Must be validated during `--setup` before bed. If the UI changes, auto-submit fails loudly with a clear error.
- **Tournament rules:** Spy bot must be valid — no external calls, responds within time limits. Probing with raises and calling to showdown are legal strategies.
