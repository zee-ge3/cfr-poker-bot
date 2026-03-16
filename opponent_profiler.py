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
