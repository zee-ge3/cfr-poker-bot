# opponent_profiler.py
"""
opponent_profiler.py — Overnight log analysis + ranked exploit report.

Usage:
  python opponent_profiler.py                     # analyze all spy-bot matches
  python opponent_profiler.py --top 20            # show top-N opponents only
  python opponent_profiler.py --min-hands 50      # lower threshold (debug)
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent
LOGS_DIR = REPO / "tournament_logs"
OVERNIGHT_LOG = REPO / "overnight_log.jsonl"
MIN_HANDS = 100  # default minimum hands for exploit notes

STREETS = ["Pre-Flop", "Flop", "Turn", "River"]
OUR_TEAM = "geoz"


# ── CSV parser ────────────────────────────────────────────────────────────────

def parse_match_csv(content: str) -> dict:
    """Parse a match CSV string. Returns structured match data.

    Returns dict with:
      geoz_slot: 0 or 1
      opp_slot: 1 - geoz_slot
      opp_name: str
      hands: list of per-hand dicts
      opp_ftr_events: list of (street, opp_is_ip, folded) dicts
        where folded=True means opp folded in response to our raise
    """
    lines = content.splitlines()
    if not lines:
        return {}

    # Parse team names from header comment
    header = lines[0]
    m = re.match(r"#\s*Team\s*0:\s*(.+?),\s*Team\s*1:\s*(.+)", header)
    if not m:
        return {}
    t0, t1 = m.group(1).strip(), m.group(2).strip()

    if OUR_TEAM.lower() in t0.lower():
        geoz_slot = 0
        opp_name = t1
    elif OUR_TEAM.lower() in t1.lower():
        geoz_slot = 1
        opp_name = t0
    else:
        return {}  # neither team is geoz

    opp_slot = 1 - geoz_slot

    # Parse data rows
    data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    if not data_lines:
        return {}

    rows = list(csv.DictReader(data_lines))
    if not rows:
        return {}

    # Group rows by hand number
    hand_rows = defaultdict(list)
    for row in rows:
        hand_rows[int(row['hand_number'])].append(row)

    hands = []
    opp_ftr_events = []  # (street, opp_is_ip, opp_folded_to_our_raise)

    for hand_num in sorted(hand_rows):
        h_rows = hand_rows[hand_num]

        # Derive opp position: first Pre-Flop actor is SB = IP post-flop
        # Heads-up rules: first Pre-Flop actor = SB, SB acts last post-flop = IP
        pf_rows = [r for r in h_rows if r['street'] == 'Pre-Flop']
        opp_is_ip = None
        if pf_rows:
            first_actor = int(pf_rows[0]['active_team'])
            opp_is_ip = (first_actor == opp_slot)  # opp is SB → IP post-flop

        hand_info = {
            'hand_num': hand_num,
            'opp_is_ip': opp_is_ip,
        }
        hands.append(hand_info)

        # Find fold-to-raise events: we raised, did opp fold? (one event per street per hand)
        for street in STREETS[1:]:  # post-flop only; Pre-Flop fold rate tracked via pf_fold_rate
            s_rows = [r for r in h_rows if r['street'] == street
                      and r['action_type'] not in ('DISCARD',)]
            # Record at most one FTR event per street per hand
            recorded_this_street = False
            for i, row in enumerate(s_rows):
                if recorded_this_street:
                    break
                actor = int(row['active_team'])
                if actor == geoz_slot and row['action_type'] == 'RAISE':
                    # Look for opp's next action
                    for j in range(i + 1, len(s_rows)):
                        next_actor = int(s_rows[j]['active_team'])
                        if next_actor == opp_slot:
                            opp_folded = s_rows[j]['action_type'] == 'FOLD'
                            opp_ftr_events.append({
                                'street': street,
                                'opp_is_ip': opp_is_ip,
                                'folded': opp_folded,
                            })
                            recorded_this_street = True
                            break

    return {
        'geoz_slot': geoz_slot,
        'opp_name': opp_name,
        'hands': hands,
        'opp_ftr_events': opp_ftr_events,
        'rows': rows,
        'opp_slot': opp_slot,
    }


# ── Metrics aggregation ────────────────────────────────────────────────────────

def aggregate_opponent(match_data_list: list) -> dict:
    """Aggregate multiple parsed match dicts into a single opponent profile.

    Args:
        match_data_list: list of dicts returned by parse_match_csv()

    Returns dict with:
        total_hands: int
        pf_fold_rate: float
        ftr_{pos}_{street}: float or nan (nan if n < 3)
        ftr_{pos}_{street}_n: int (sample count)
      where pos in ('oop', 'ip'), street in STREETS[1:]
    """
    total_hands = 0
    ftr_counts = defaultdict(lambda: {'fold': 0, 'total': 0})
    pf_fold_total = 0
    pf_action_total = 0

    for md in match_data_list:
        if not md:
            continue
        total_hands += len(md.get('hands', []))

        # FTR events
        for ev in md.get('opp_ftr_events', []):
            pos = 'ip' if ev['opp_is_ip'] else 'oop'
            key = f"{pos}_{ev['street']}"
            ftr_counts[key]['total'] += 1
            if ev['folded']:
                ftr_counts[key]['fold'] += 1

        # Pre-flop fold rate from raw rows
        opp_slot = md.get('opp_slot', 0)
        for row in md.get('rows', []):
            if row['street'] == 'Pre-Flop' and int(row['active_team']) == opp_slot:
                pf_action_total += 1
                if row['action_type'] == 'FOLD':
                    pf_fold_total += 1

    # Build profile
    import math
    profile = {'total_hands': total_hands}
    profile['pf_fold_rate'] = pf_fold_total / pf_action_total if pf_action_total else 0.0

    for pos in ('oop', 'ip'):
        for street in STREETS[1:]:  # post-flop only, matching parse_match_csv
            key = f"{pos}_{street}"
            d = ftr_counts[key]
            # Require n >= 3 for a meaningful rate; return nan otherwise
            val = d['fold'] / d['total'] if d['total'] >= 3 else float('nan')
            profile[f"ftr_{key}"] = val
            profile[f"ftr_{key}_n"] = d['total']

    return profile


# ── Opponent type classification ───────────────────────────────────────────────

def classify_opponent_type(pf_fold_rate: float, vpip: float,
                           raise_rate: float = 0.5) -> str:
    """Classify opponent into tag/lag/maniac/calling_station.

    Args:
        pf_fold_rate: fraction of pre-flop actions where opponent folds
        vpip: voluntarily put money in pot rate (1 - pf_fold_rate proxy)
        raise_rate: fraction of actions that are raises (default 0.5)

    Returns one of: 'calling_station', 'maniac', 'lag', 'tag'
    """
    if vpip >= 0.80 and raise_rate < 0.15:
        return 'calling_station'
    if pf_fold_rate <= 0.15:
        return 'maniac'
    if pf_fold_rate <= 0.35:
        return 'lag'
    return 'tag'


def generate_exploit_note(opp_name: str, rank: int, profile: dict) -> str:
    """Generate a one-line exploit note from an opponent profile.

    Args:
        opp_name: opponent team name
        rank: current leaderboard rank (or None)
        profile: dict from aggregate_opponent()

    Returns formatted exploit note string.
    """
    import math
    notes = []

    # High OOP fold-to-raise (post-flop) — biggest exploit signal
    for street in ('Flop', 'Turn', 'River'):
        rate = profile.get(f'ftr_oop_{street}', float('nan'))
        n = profile.get(f'ftr_oop_{street}_n', 0)
        if n >= 10 and not math.isnan(rate) and rate >= 0.65:
            notes.append(f"folds {rate:.0%} OOP on {street} → raise every {street} when they're OOP")
            break

    # High IP fold-to-raise
    for street in ('Flop', 'Turn'):
        rate = profile.get(f'ftr_ip_{street}', float('nan'))
        n = profile.get(f'ftr_ip_{street}_n', 0)
        if n >= 10 and not math.isnan(rate) and rate >= 0.55:
            notes.append(f"also folds {rate:.0%} IP on {street} → probe aggressively")
            break

    # Very low PF fold → blind steals useless
    pf = profile.get('pf_fold_rate', 0.5)
    if not math.isnan(pf) and pf < 0.10:
        notes.append("never folds PF → skip blind steals, raise post-flop instead")

    if not notes:
        notes.append("no strong exploit signal yet")

    rank_str = f"(#{rank})" if rank is not None else ""
    return f"{opp_name} {rank_str}: " + "; ".join(notes)
