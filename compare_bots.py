#!/usr/bin/env python3
"""
compare_bots.py — Comprehensive comparison of PlayerAgent vs Claude2Agent
tournament performance from CSV logs.
"""

import csv
import os
import re
import sys
from collections import defaultdict

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_logs")
OUR_TEAM = "geoz"

# Street name to numeric mapping
STREET_NUM = {"Pre-Flop": 0, "Flop": 1, "Turn": 2, "River": 3}


def detect_geoz_slot(content):
    """Returns (slot, geoz_team_name, opponent_name) from CSV comment header."""
    for line in content.splitlines():
        if not line.startswith("#"):
            break
        m = re.match(r"#\s*Team 0:\s*(.+?)\s*,\s*Team 1:\s*(.+)", line)
        if m:
            t0, t1 = m.group(1).strip(), m.group(2).strip()
            if OUR_TEAM.lower() in t0.lower():
                return 0, t0, t1
            if OUR_TEAM.lower() in t1.lower():
                return 1, t1, t0
    return None, None, None


def detect_bot_type_for_old_csv(match_id):
    """For old-format match_NNNN.csv files, check log files to determine bot type."""
    # Check for explicit bot logs
    for fname in os.listdir(LOGS_DIR):
        if not fname.startswith(f"match_{match_id}_bot_"):
            continue
        if fname.endswith(".log"):
            bot_name = fname.replace(f"match_{match_id}_bot_", "").replace(".log", "")
            if bot_name == "Claude2Agent":
                return "Claude2Agent"
            elif bot_name == "PlayerAgent":
                return "PlayerAgent"
            elif bot_name.startswith("AEA"):
                return "PlayerAgent"
            elif bot_name.startswith("AVB") or bot_name.startswith("autowin"):
                return "PlayerAgent"
            else:
                return "PlayerAgent"  # default old bots to PlayerAgent
    return None


def classify_files():
    """Classify all CSVs into PlayerAgent or Claude2Agent matches.
    Returns dict: bot_type -> list of (match_id, csv_path)
    """
    result = {"PlayerAgent": [], "Claude2Agent": []}

    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(LOGS_DIR, fname)

        # Check suffixed files first
        m = re.match(r"match_(\d+)_Claude2Agent\.csv$", fname)
        if m:
            result["Claude2Agent"].append((int(m.group(1)), fpath))
            continue

        m = re.match(r"match_(\d+)_PlayerAgent\.csv$", fname)
        if m:
            result["PlayerAgent"].append((int(m.group(1)), fpath))
            continue

        # Old-format: match_NNNN.csv
        m = re.match(r"match_(\d+)\.csv$", fname)
        if m:
            match_id = int(m.group(1))
            bot_type = detect_bot_type_for_old_csv(match_id)
            if bot_type and bot_type in result:
                result[bot_type].append((match_id, fpath))
            # Skip if we can't determine bot type

    return result


def parse_csv(fpath):
    """Parse a CSV file. Returns (rows, geoz_slot, opponent_name, content) or None."""
    try:
        with open(fpath) as f:
            content = f.read()
    except Exception:
        return None

    geoz_slot, _, opp_name = detect_geoz_slot(content)
    if geoz_slot is None:
        return None

    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    if not data_lines:
        return None

    rows = list(csv.DictReader(data_lines))
    if not rows:
        return None

    return rows, geoz_slot, opp_name, content


def analyze_match(rows, geoz_slot):
    """Deep analysis of a single match. Returns a dict of stats."""
    opp_slot = 1 - geoz_slot
    our_col = f"team_{geoz_slot}_bankroll"
    opp_col = f"team_{opp_slot}_bankroll"
    our_bet_col = f"team_{geoz_slot}_bet"
    opp_bet_col = f"team_{opp_slot}_bet"

    stats = {
        "final_bankroll": 0,
        "total_hands": 0,
        "actions": {"FOLD": 0, "CALL": 0, "RAISE": 0, "CHECK": 0, "DISCARD": 0},
        "opp_actions": {"FOLD": 0, "CALL": 0, "RAISE": 0, "CHECK": 0, "DISCARD": 0},
        "preflop_folds": 0,
        "preflop_actions": 0,
        "postflop_fold_to_raise": 0,
        "postflop_opp_raise_then_geoz_acts": 0,
        "geoz_raises_by_street": defaultdict(int),
        "opp_raises_by_street": defaultdict(int),
        "geoz_actions_by_street": defaultdict(int),
        "opp_actions_by_street": defaultdict(int),
        "showdown_net": 0,
        "fold_net": 0,
        "showdown_count": 0,
        "fold_count": 0,
        "showdown_pot_sizes": [],
        "fold_chip_losses": [],
        "bankroll_at_250": None,
        "bankroll_at_500": None,
        "bankroll_at_750": None,
    }

    # Get final bankroll
    last_row = rows[-1]
    stats["final_bankroll"] = float(last_row.get(our_col, 0))
    stats["total_hands"] = int(last_row.get("hand_number", 0)) + 1

    # Group rows by hand
    hands = defaultdict(list)
    for row in rows:
        h = int(row["hand_number"])
        hands[h].append(row)

    # Track bankroll at specific hands
    hand_bankrolls = {}
    for row in rows:
        h = int(row["hand_number"])
        hand_bankrolls[h] = float(row.get(our_col, 0))

    for target in [250, 500, 750]:
        if target in hand_bankrolls:
            stats[f"bankroll_at_{target}"] = hand_bankrolls[target]
        else:
            # Find closest hand <= target
            closest = None
            for h in sorted(hand_bankrolls.keys()):
                if h <= target:
                    closest = h
            if closest is not None:
                stats[f"bankroll_at_{target}"] = hand_bankrolls[closest]

    # Analyze each hand
    for h_num in sorted(hands.keys()):
        hand_rows = hands[h_num]
        prev_action = None
        prev_player = None

        hand_ended_fold = False
        hand_ended_showdown = False

        # Track bankroll at start and end of hand
        bankroll_start = float(hand_rows[0].get(our_col, 0))

        # Find last non-DISCARD action to determine hand ending
        non_discard_actions = [r for r in hand_rows
                               if r.get("action_type") not in ("DISCARD", None, "")]
        if non_discard_actions:
            last_action = non_discard_actions[-1]
            if last_action.get("action_type") == "FOLD":
                hand_ended_fold = True
            else:
                # Check if hand reaches River or ends with call/check
                streets_seen = set()
                for r in hand_rows:
                    s = r.get("street", "")
                    if s in STREET_NUM:
                        streets_seen.add(STREET_NUM[s])
                if 3 in streets_seen:  # River was reached
                    last_street_actions = [r for r in hand_rows
                                           if r.get("street") == "River"
                                           and r.get("action_type") not in ("DISCARD", None, "")]
                    if last_street_actions:
                        la = last_street_actions[-1].get("action_type")
                        if la in ("CALL", "CHECK"):
                            hand_ended_showdown = True

        for row in hand_rows:
            action = row.get("action_type", "")
            if action in ("", "DISCARD"):
                continue

            active = int(row.get("active_team", -1))
            street = row.get("street", "")
            street_num = STREET_NUM.get(street, -1)

            is_geoz = (active == geoz_slot)

            if is_geoz:
                if action in stats["actions"]:
                    stats["actions"][action] += 1
                if street_num == 0:
                    stats["preflop_actions"] += 1
                    if action == "FOLD":
                        stats["preflop_folds"] += 1
                if action == "RAISE":
                    stats["geoz_raises_by_street"][street_num] += 1
                stats["geoz_actions_by_street"][street_num] += 1

                # Post-flop fold-to-raise
                if street_num > 0 and prev_action == "RAISE" and prev_player == opp_slot:
                    stats["postflop_opp_raise_then_geoz_acts"] += 1
                    if action == "FOLD":
                        stats["postflop_fold_to_raise"] += 1
            else:
                if action in stats["opp_actions"]:
                    stats["opp_actions"][action] += 1
                if action == "RAISE":
                    stats["opp_raises_by_street"][street_num] += 1
                stats["opp_actions_by_street"][street_num] += 1

            prev_action = action
            prev_player = active

        # Chip flow
        # Get bankroll at end of this hand (start of next hand, or final)
        next_hands = [hh for hh in sorted(hands.keys()) if hh > h_num]
        if next_hands:
            bankroll_end = float(hands[next_hands[0]][0].get(our_col, 0))
        else:
            # Last hand - use final value from last row
            bankroll_end = float(hand_rows[-1].get(our_col, 0))
            # For the last hand we need the actual result
            # The bankroll in rows shows the bankroll at the START of the hand
            # After the last action, the bankroll changes
            # We'll estimate from the bet columns
            # Actually the bankroll columns show cumulative net, not per-hand
            # Let's recalculate

        pnl = bankroll_end - bankroll_start

        if hand_ended_fold:
            stats["fold_net"] += pnl
            stats["fold_count"] += 1
            if pnl < 0:
                stats["fold_chip_losses"].append(abs(pnl))
        elif hand_ended_showdown:
            stats["showdown_net"] += pnl
            stats["showdown_count"] += 1
            # Pot size at showdown: our_bet + opp_bet from last row
            last_r = hand_rows[-1]
            try:
                our_bet = float(last_r.get(our_bet_col, 0) or 0)
                opp_bet = float(last_r.get(opp_bet_col, 0) or 0)
                # Actually the bet columns show current bet amounts
                # Total pot = our_bet + opp_bet at end of hand
                pot = our_bet + opp_bet
                if pot > 0:
                    stats["showdown_pot_sizes"].append(pot)
            except (ValueError, TypeError):
                pass

    return stats


def detect_autofolding(rows, geoz_slot):
    """Detect if autofolding occurred at end of match.
    Returns True if the last 20+ hands show geoz folding every hand.
    """
    hands = defaultdict(list)
    for row in rows:
        h = int(row["hand_number"])
        hands[h].append(row)

    sorted_hands = sorted(hands.keys(), reverse=True)
    consecutive_folds = 0

    for h_num in sorted_hands:
        hand_rows = hands[h_num]
        geoz_actions = [r for r in hand_rows
                        if int(r.get("active_team", -1)) == geoz_slot
                        and r.get("action_type") not in ("DISCARD", None, "")]
        if geoz_actions:
            # Check if geoz's first (and possibly only) action is FOLD
            first_action = geoz_actions[0].get("action_type")
            if first_action == "FOLD":
                consecutive_folds += 1
            else:
                break
        else:
            break

    return consecutive_folds >= 20


def print_separator(char="=", width=90):
    print(char * width)


def print_header(title, width=90):
    print()
    print_separator("=", width)
    print(f"  {title}")
    print_separator("=", width)


def print_subheader(title, width=90):
    print()
    print(f"  --- {title} ---")


def format_pct(num, denom):
    if denom == 0:
        return "  N/A"
    return f"{num / denom * 100:5.1f}%"


def main():
    if not os.path.isdir(LOGS_DIR):
        print(f"Error: {LOGS_DIR} not found")
        sys.exit(1)

    # Classify files
    classified = classify_files()

    print_header("POKER BOT TOURNAMENT COMPARISON: PlayerAgent vs Claude2Agent")
    print(f"  Team: {OUR_TEAM}")
    print(f"  Log directory: {LOGS_DIR}")
    print(f"  PlayerAgent matches found: {len(classified['PlayerAgent'])}")
    print(f"  Claude2Agent matches found: {len(classified['Claude2Agent'])}")

    # Parse all matches
    bot_data = {}  # bot_type -> list of (match_id, stats, opp_name, rows, geoz_slot)

    for bot_type in ("PlayerAgent", "Claude2Agent"):
        bot_data[bot_type] = []
        for match_id, fpath in classified[bot_type]:
            parsed = parse_csv(fpath)
            if parsed is None:
                continue
            rows, geoz_slot, opp_name, content = parsed
            stats = analyze_match(rows, geoz_slot)
            bot_data[bot_type].append({
                "match_id": match_id,
                "stats": stats,
                "opp_name": opp_name,
                "rows": rows,
                "geoz_slot": geoz_slot,
            })

    # =========================================================================
    # SECTION 1: Win/Loss Record
    # =========================================================================
    for bot_type in ("PlayerAgent", "Claude2Agent"):
        matches = bot_data[bot_type]
        print_header(f"1. WIN/LOSS RECORD — {bot_type}")

        if not matches:
            print("  No matches found.")
            continue

        wins = [m for m in matches if m["stats"]["final_bankroll"] > 0]
        losses = [m for m in matches if m["stats"]["final_bankroll"] <= 0]

        win_margins = [m["stats"]["final_bankroll"] for m in wins]
        loss_margins = [m["stats"]["final_bankroll"] for m in losses]

        avg_win = sum(win_margins) / len(win_margins) if win_margins else 0
        avg_loss = sum(loss_margins) / len(loss_margins) if loss_margins else 0

        print(f"  Total matches:           {len(matches)}")
        print(f"  Wins:                    {len(wins)}")
        print(f"  Losses:                  {len(losses)}")
        print(f"  Win rate:                {len(wins) / len(matches) * 100:.1f}%")
        print(f"  Avg margin of victory:   {avg_win:+.0f}")
        print(f"  Avg margin of defeat:    {avg_loss:+.0f}")
        print(f"  Total net chips:         {sum(m['stats']['final_bankroll'] for m in matches):+.0f}")

        print_subheader("Match-by-match detail")
        print(f"  {'Match':>7}  {'Opponent':<25}  {'Final':>8}  {'W/L':>3}  {'Hands':>5}")
        print(f"  {'─'*7}  {'─'*25}  {'─'*8}  {'─'*3}  {'─'*5}")

        for m in sorted(matches, key=lambda x: x["match_id"]):
            fb = m["stats"]["final_bankroll"]
            wl = "W" if fb > 0 else "L"
            hands = m["stats"]["total_hands"]
            print(f"  {m['match_id']:>7}  {m['opp_name']:<25}  {fb:>+8.0f}  {wl:>3}  {hands:>5}")

    # =========================================================================
    # SECTION 2: Action Profile
    # =========================================================================
    for bot_type in ("PlayerAgent", "Claude2Agent"):
        matches = bot_data[bot_type]
        print_header(f"2. ACTION PROFILE — {bot_type}")

        if not matches:
            print("  No matches found.")
            continue

        # Aggregate actions
        total_actions = defaultdict(int)
        total_opp_actions = defaultdict(int)
        total_preflop_folds = 0
        total_preflop_actions = 0
        total_pf_fold_to_raise = 0
        total_pf_opp_raise_geoz_acts = 0
        geoz_raises_street = defaultdict(int)
        opp_raises_street = defaultdict(int)
        geoz_acts_street = defaultdict(int)
        opp_acts_street = defaultdict(int)

        for m in matches:
            s = m["stats"]
            for a, c in s["actions"].items():
                total_actions[a] += c
            for a, c in s["opp_actions"].items():
                total_opp_actions[a] += c
            total_preflop_folds += s["preflop_folds"]
            total_preflop_actions += s["preflop_actions"]
            total_pf_fold_to_raise += s["postflop_fold_to_raise"]
            total_pf_opp_raise_geoz_acts += s["postflop_opp_raise_then_geoz_acts"]
            for st, c in s["geoz_raises_by_street"].items():
                geoz_raises_street[st] += c
            for st, c in s["opp_raises_by_street"].items():
                opp_raises_street[st] += c
            for st, c in s["geoz_actions_by_street"].items():
                geoz_acts_street[st] += c
            for st, c in s["opp_actions_by_street"].items():
                opp_acts_street[st] += c

        total_geoz_all = sum(total_actions.values())
        total_opp_all = sum(total_opp_actions.values())

        print_subheader("geoz action totals")
        print(f"  {'Action':<10} {'Count':>7}  {'% of total':>10}")
        print(f"  {'─'*10} {'─'*7}  {'─'*10}")
        for action in ("FOLD", "CALL", "RAISE", "CHECK"):
            c = total_actions[action]
            pct = c / total_geoz_all * 100 if total_geoz_all else 0
            print(f"  {action:<10} {c:>7}  {pct:>9.1f}%")
        print(f"  {'TOTAL':<10} {total_geoz_all:>7}")

        print_subheader("Key rates")
        print(f"  Pre-flop fold rate:          {format_pct(total_preflop_folds, total_preflop_actions)}"
              f"  ({total_preflop_folds}/{total_preflop_actions})")
        print(f"  Post-flop fold-to-raise:     {format_pct(total_pf_fold_to_raise, total_pf_opp_raise_geoz_acts)}"
              f"  ({total_pf_fold_to_raise}/{total_pf_opp_raise_geoz_acts})")

        print_subheader("Raise frequency by street (geoz vs opponent)")
        street_names = {0: "Pre-Flop", 1: "Flop", 2: "Turn", 3: "River"}
        print(f"  {'Street':<10} {'geoz raises':>12} {'geoz rate':>10}  {'opp raises':>11} {'opp rate':>10}")
        print(f"  {'─'*10} {'─'*12} {'─'*10}  {'─'*11} {'─'*10}")
        for st in range(4):
            gr = geoz_raises_street.get(st, 0)
            ga = geoz_acts_street.get(st, 0)
            opr = opp_raises_street.get(st, 0)
            opa = opp_acts_street.get(st, 0)
            g_rate = format_pct(gr, ga)
            o_rate = format_pct(opr, opa)
            print(f"  {street_names[st]:<10} {gr:>12} {g_rate:>10}  {opr:>11} {o_rate:>10}")

    # =========================================================================
    # SECTION 3: Chip Flow Analysis
    # =========================================================================
    for bot_type in ("PlayerAgent", "Claude2Agent"):
        matches = bot_data[bot_type]
        print_header(f"3. CHIP FLOW ANALYSIS — {bot_type}")

        if not matches:
            print("  No matches found.")
            continue

        total_showdown_net = 0
        total_fold_net = 0
        total_showdown_count = 0
        total_fold_count = 0
        all_showdown_pots = []
        all_fold_losses = []

        for m in matches:
            s = m["stats"]
            total_showdown_net += s["showdown_net"]
            total_fold_net += s["fold_net"]
            total_showdown_count += s["showdown_count"]
            total_fold_count += s["fold_count"]
            all_showdown_pots.extend(s["showdown_pot_sizes"])
            all_fold_losses.extend(s["fold_chip_losses"])

        avg_showdown_pot = sum(all_showdown_pots) / len(all_showdown_pots) if all_showdown_pots else 0
        avg_fold_loss = sum(all_fold_losses) / len(all_fold_losses) if all_fold_losses else 0

        print(f"  Net chips from showdowns:    {total_showdown_net:>+10.0f}  ({total_showdown_count} showdowns)")
        print(f"  Net chips from folds:        {total_fold_net:>+10.0f}  ({total_fold_count} fold-ending hands)")
        print(f"  Average pot at showdown:     {avg_showdown_pot:>10.1f}")
        print(f"  Average chips lost per fold:  {avg_fold_loss:>10.1f}")

        if total_showdown_count > 0:
            print(f"  Avg chips/showdown:          {total_showdown_net / total_showdown_count:>+10.1f}")
        if total_fold_count > 0:
            print(f"  Avg chips/fold-hand:         {total_fold_net / total_fold_count:>+10.1f}")

    # =========================================================================
    # SECTION 4: Per-Opponent Breakdown
    # =========================================================================
    for bot_type in ("PlayerAgent", "Claude2Agent"):
        matches = bot_data[bot_type]
        print_header(f"4. PER-OPPONENT BREAKDOWN — {bot_type}")

        if not matches:
            print("  No matches found.")
            continue

        by_opp = defaultdict(lambda: {"wins": 0, "losses": 0, "net": 0, "matches": []})
        for m in matches:
            opp = m["opp_name"]
            fb = m["stats"]["final_bankroll"]
            if fb > 0:
                by_opp[opp]["wins"] += 1
            else:
                by_opp[opp]["losses"] += 1
            by_opp[opp]["net"] += fb
            by_opp[opp]["matches"].append(m["match_id"])

        print(f"  {'Opponent':<25} {'W':>3}-{'L':<3} {'WR':>6}  {'Net':>8}  {'Matches':>7}  Match IDs")
        print(f"  {'─'*25} {'─'*3}-{'─'*3} {'─'*6}  {'─'*8}  {'─'*7}  {'─'*20}")

        for opp, d in sorted(by_opp.items(), key=lambda x: x[1]["net"], reverse=True):
            t = d["wins"] + d["losses"]
            wr = d["wins"] / t * 100 if t else 0
            ids = ",".join(str(x) for x in sorted(d["matches"])[:5])
            if len(d["matches"]) > 5:
                ids += "..."
            print(f"  {opp:<25} {d['wins']:>3}-{d['losses']:<3} {wr:>5.1f}%  {d['net']:>+8.0f}  {t:>7}  {ids}")

    # =========================================================================
    # SECTION 5: Head-to-Head Comparison
    # =========================================================================
    print_header("5. HEAD-TO-HEAD COMPARISON: PlayerAgent vs Claude2Agent")

    # Find common opponents
    pa_opps = set()
    c2_opps = set()
    for m in bot_data["PlayerAgent"]:
        pa_opps.add(m["opp_name"])
    for m in bot_data["Claude2Agent"]:
        c2_opps.add(m["opp_name"])

    common = pa_opps & c2_opps
    pa_only = pa_opps - c2_opps
    c2_only = c2_opps - pa_opps

    print(f"  Common opponents:        {len(common)}")
    print(f"  PlayerAgent-only opps:   {len(pa_only)}")
    print(f"  Claude2Agent-only opps:  {len(c2_only)}")

    if common:
        print_subheader("Performance vs common opponents")
        print(f"  {'Opponent':<25} {'PA W-L':>7} {'PA Net':>8}  {'C2 W-L':>7} {'C2 Net':>8}  {'Better':>8}")
        print(f"  {'─'*25} {'─'*7} {'─'*8}  {'─'*7} {'─'*8}  {'─'*8}")

        for opp in sorted(common):
            pa_w = sum(1 for m in bot_data["PlayerAgent"] if m["opp_name"] == opp and m["stats"]["final_bankroll"] > 0)
            pa_l = sum(1 for m in bot_data["PlayerAgent"] if m["opp_name"] == opp and m["stats"]["final_bankroll"] <= 0)
            pa_net = sum(m["stats"]["final_bankroll"] for m in bot_data["PlayerAgent"] if m["opp_name"] == opp)

            c2_w = sum(1 for m in bot_data["Claude2Agent"] if m["opp_name"] == opp and m["stats"]["final_bankroll"] > 0)
            c2_l = sum(1 for m in bot_data["Claude2Agent"] if m["opp_name"] == opp and m["stats"]["final_bankroll"] <= 0)
            c2_net = sum(m["stats"]["final_bankroll"] for m in bot_data["Claude2Agent"] if m["opp_name"] == opp)

            better = "PA" if pa_net > c2_net else ("C2" if c2_net > pa_net else "TIE")
            print(f"  {opp:<25} {pa_w}-{pa_l:>3} {pa_net:>+8.0f}  {c2_w}-{c2_l:>3} {c2_net:>+8.0f}  {better:>8}")

    # Aggregate comparisons
    print_subheader("Aggregate behavior comparison")

    for metric_name, get_metric in [
        ("Post-flop fold-to-raise rate", lambda ms: (
            sum(m["stats"]["postflop_fold_to_raise"] for m in ms),
            sum(m["stats"]["postflop_opp_raise_then_geoz_acts"] for m in ms)
        )),
        ("Total raise count", lambda ms: (
            sum(m["stats"]["actions"]["RAISE"] for m in ms), 1
        )),
        ("Raise % of actions", lambda ms: (
            sum(m["stats"]["actions"]["RAISE"] for m in ms),
            sum(sum(m["stats"]["actions"].values()) for m in ms)
        )),
    ]:
        pa_num, pa_den = get_metric(bot_data["PlayerAgent"])
        c2_num, c2_den = get_metric(bot_data["Claude2Agent"])

        if metric_name == "Total raise count":
            pa_val = f"{pa_num}"
            c2_val = f"{c2_num}"
            # Per-match average
            pa_pm = pa_num / len(bot_data["PlayerAgent"]) if bot_data["PlayerAgent"] else 0
            c2_pm = c2_num / len(bot_data["Claude2Agent"]) if bot_data["Claude2Agent"] else 0
            print(f"  {metric_name:<35} PA: {pa_val:>8}  (avg {pa_pm:.0f}/match)    "
                  f"C2: {c2_val:>8}  (avg {c2_pm:.0f}/match)")
        else:
            pa_pct = pa_num / pa_den * 100 if pa_den else 0
            c2_pct = c2_num / c2_den * 100 if c2_den else 0
            print(f"  {metric_name:<35} PA: {pa_pct:>7.1f}%    C2: {c2_pct:>7.1f}%")

    # Showdown comparison
    pa_sd_net = sum(m["stats"]["showdown_net"] for m in bot_data["PlayerAgent"])
    pa_sd_cnt = sum(m["stats"]["showdown_count"] for m in bot_data["PlayerAgent"])
    c2_sd_net = sum(m["stats"]["showdown_net"] for m in bot_data["Claude2Agent"])
    c2_sd_cnt = sum(m["stats"]["showdown_count"] for m in bot_data["Claude2Agent"])

    pa_sd_avg = pa_sd_net / pa_sd_cnt if pa_sd_cnt else 0
    c2_sd_avg = c2_sd_net / c2_sd_cnt if c2_sd_cnt else 0

    print(f"  {'Showdown net chips':<35} PA: {pa_sd_net:>+8.0f} ({pa_sd_cnt} SDs, avg {pa_sd_avg:+.1f})    "
          f"C2: {c2_sd_net:>+8.0f} ({c2_sd_cnt} SDs, avg {c2_sd_avg:+.1f})")

    # Overall verdict
    pa_matches = bot_data["PlayerAgent"]
    c2_matches = bot_data["Claude2Agent"]
    pa_wr = len([m for m in pa_matches if m["stats"]["final_bankroll"] > 0]) / len(pa_matches) * 100 if pa_matches else 0
    c2_wr = len([m for m in c2_matches if m["stats"]["final_bankroll"] > 0]) / len(c2_matches) * 100 if c2_matches else 0
    pa_net = sum(m["stats"]["final_bankroll"] for m in pa_matches)
    c2_net = sum(m["stats"]["final_bankroll"] for m in c2_matches)
    pa_avg_net = pa_net / len(pa_matches) if pa_matches else 0
    c2_avg_net = c2_net / len(c2_matches) if c2_matches else 0

    print_subheader("Overall verdict")
    print(f"  {'Metric':<30} {'PlayerAgent':>15} {'Claude2Agent':>15}")
    print(f"  {'─'*30} {'─'*15} {'─'*15}")
    print(f"  {'Win rate':<30} {pa_wr:>14.1f}% {c2_wr:>14.1f}%")
    print(f"  {'Total net chips':<30} {pa_net:>+15.0f} {c2_net:>+15.0f}")
    print(f"  {'Avg net per match':<30} {pa_avg_net:>+15.0f} {c2_avg_net:>+15.0f}")
    print(f"  {'Matches played':<30} {len(pa_matches):>15} {len(c2_matches):>15}")

    # =========================================================================
    # SECTION 6: Claude2Agent Loss Detail
    # =========================================================================
    print_header("6. CLAUDE2AGENT LOSS DETAIL")

    c2_losses = [m for m in bot_data["Claude2Agent"] if m["stats"]["final_bankroll"] <= 0]

    if not c2_losses:
        print("  No Claude2Agent losses found.")
    else:
        print(f"  Total Claude2Agent losses: {len(c2_losses)}")
        print()

        print(f"  {'Match':>7}  {'Opponent':<25}  {'Final':>8}  {'PF FtR':>7}  {'geoz R':>7}  {'opp R':>7}  "
              f"{'AutoF':>5}  {'@250':>6}  {'@500':>6}  {'@750':>6}")
        print(f"  {'─'*7}  {'─'*25}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")

        for m in sorted(c2_losses, key=lambda x: x["match_id"]):
            s = m["stats"]
            fb = s["final_bankroll"]
            opp = m["opp_name"]

            # Post-flop fold-to-raise for this match
            pf_ftr_num = s["postflop_fold_to_raise"]
            pf_ftr_den = s["postflop_opp_raise_then_geoz_acts"]
            pf_ftr = f"{pf_ftr_num}/{pf_ftr_den}" if pf_ftr_den > 0 else "N/A"

            geoz_raises = s["actions"]["RAISE"]
            opp_raises = s["opp_actions"]["RAISE"]

            # Autofolding detection
            autofold = detect_autofolding(m["rows"], m["geoz_slot"])
            af_str = "YES" if autofold else "no"

            b250 = f"{s['bankroll_at_250']:>+6.0f}" if s["bankroll_at_250"] is not None else "  N/A"
            b500 = f"{s['bankroll_at_500']:>+6.0f}" if s["bankroll_at_500"] is not None else "  N/A"
            b750 = f"{s['bankroll_at_750']:>+6.0f}" if s["bankroll_at_750"] is not None else "  N/A"

            print(f"  {m['match_id']:>7}  {opp:<25}  {fb:>+8.0f}  {pf_ftr:>7}  {geoz_raises:>7}  {opp_raises:>7}  "
                  f"{af_str:>5}  {b250}  {b500}  {b750}")

        # Detailed analysis per loss
        print()
        print_subheader("Detailed loss narratives")
        for m in sorted(c2_losses, key=lambda x: x["stats"]["final_bankroll"]):
            s = m["stats"]
            print(f"\n  Match {m['match_id']} vs {m['opp_name']}:")
            print(f"    Final bankroll: {s['final_bankroll']:+.0f}")
            print(f"    Total hands: {s['total_hands']}")

            pf_ftr_num = s["postflop_fold_to_raise"]
            pf_ftr_den = s["postflop_opp_raise_then_geoz_acts"]
            if pf_ftr_den > 0:
                print(f"    Post-flop fold-to-raise: {pf_ftr_num}/{pf_ftr_den} = {pf_ftr_num/pf_ftr_den*100:.1f}%")
            else:
                print(f"    Post-flop fold-to-raise: N/A (opp never raised post-flop)")

            print(f"    geoz raises: {s['actions']['RAISE']}, opponent raises: {s['opp_actions']['RAISE']}")

            autofold = detect_autofolding(m["rows"], m["geoz_slot"])
            if autofold:
                print(f"    AUTOFOLDING DETECTED: geoz folded many consecutive hands at end of match")

            # Trajectory
            trajectory = []
            if s["bankroll_at_250"] is not None:
                trajectory.append(f"@250={s['bankroll_at_250']:+.0f}")
            if s["bankroll_at_500"] is not None:
                trajectory.append(f"@500={s['bankroll_at_500']:+.0f}")
            if s["bankroll_at_750"] is not None:
                trajectory.append(f"@750={s['bankroll_at_750']:+.0f}")
            trajectory.append(f"final={s['final_bankroll']:+.0f}")
            print(f"    Trajectory: {', '.join(trajectory)}")

            # Shape analysis
            if s["bankroll_at_500"] is not None:
                mid = s["bankroll_at_500"]
                final = s["final_bankroll"]
                if mid > 0 and final < -200:
                    print(f"    Pattern: COLLAPSE (winning at midpoint, heavy loss at end)")
                elif mid < -200:
                    print(f"    Pattern: EARLY BLEED (already losing heavily by midpoint)")
                elif abs(final) < 100:
                    print(f"    Pattern: CLOSE LOSS (narrow margin)")
                else:
                    print(f"    Pattern: GRADUAL DECLINE")

    print()
    print_separator("=")
    print("  END OF REPORT")
    print_separator("=")


if __name__ == "__main__":
    main()
