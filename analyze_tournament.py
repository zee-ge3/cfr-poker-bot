#!/usr/bin/env python3
"""Comprehensive tournament analysis by bot version."""

import csv
import glob
import os
import re
from collections import defaultdict

LOG_DIR = '/home/g30rgez/poker/poker-engine-2026/tournament_logs'

# === BOT FAMILY MAPPING ===
# Map raw bot names to families
def bot_family(name):
    """Map a bot log name to its family."""
    if name in ('AEA', 'PlayerAgent', 'AEAv16', 'AEAv17_timing', 'AEAv18'):
        return 'PlayerAgent'
    if name in ('autowin', 'autowin_v2', 'autowin_v3', 'AVB', 'Claude2Agent',
                'AVBv6scaled', 'AVBv6scaled_v2', 'AVBv6.2_v2', 'AVBv6.3_bugfix', 'AVBv10'):
        return 'Claude2Agent'
    return name  # unknown

# === STEP 1: Build match_id -> bot_name mapping from .log files ===
def build_bot_map():
    """Map match IDs to bot names from .log filenames."""
    bot_map = {}
    log_files = glob.glob(os.path.join(LOG_DIR, 'match_*_bot_*.log'))
    for lf in log_files:
        base = os.path.basename(lf)
        # match_{id}_bot_{botName}.log
        m = re.match(r'match_(\d+)_bot_(.+)\.log', base)
        if m:
            match_id = int(m.group(1))
            bot_name = m.group(2)
            bot_map[match_id] = bot_name
    return bot_map

# === STEP 2: Parse a CSV to extract match results ===
def parse_csv(filepath):
    """Parse a match CSV and return match info dict."""
    with open(filepath) as fh:
        header_line = fh.readline().strip()
        # Parse team names from comment: # Team 0: XXX, Team 1: YYY
        m = re.match(r'#\s*Team\s*0:\s*(.+?),\s*Team\s*1:\s*(.+)', header_line)
        if not m:
            return None
        team0_name = m.group(1).strip()
        team1_name = m.group(2).strip()

        reader = csv.DictReader(fh)
        rows = list(reader)
        if not rows:
            return None

    # Determine which team is geoz
    if 'geoz' in team0_name.lower():
        geoz_team = 0
        opp_name = team1_name
    elif 'geoz' in team1_name.lower():
        geoz_team = 1
        opp_name = team0_name
    else:
        return None  # geoz not in this match

    # Build hand-level data
    # Group rows by hand number
    hand_first_row = {}
    hand_last_row = {}
    hand_rows_map = defaultdict(list)
    for r in rows:
        h = int(r['hand_number'])
        hand_rows_map[h].append(r)
        if h not in hand_first_row:
            hand_first_row[h] = r
        hand_last_row[h] = r

    hands = sorted(hand_first_row.keys())
    total_hands = len(hands)

    # Compute per-hand results for geoz using bankroll transitions
    hand_deltas = []  # (hand_number, delta_for_geoz)

    for i in range(len(hands) - 1):
        h = hands[i]
        h_next = hands[i + 1]
        if geoz_team == 1:
            b_start = int(hand_first_row[h]['team_1_bankroll'])
            b_next = int(hand_first_row[h_next]['team_1_bankroll'])
        else:
            b_start = int(hand_first_row[h]['team_0_bankroll'])
            b_next = int(hand_first_row[h_next]['team_0_bankroll'])
        delta = b_next - b_start
        hand_deltas.append((h, delta))

    # Handle last hand
    last_h = hands[-1]
    last_rows = hand_rows_map[last_h]
    last_row = last_rows[-1]
    last_action = last_row['action_type']
    last_active = int(last_row['active_team'])
    last_bet0 = int(last_row['team_0_bet'])
    last_bet1 = int(last_row['team_1_bet'])

    if last_action == 'FOLD':
        # The team that folded loses; the other team wins the pot
        # Pot is the sum of current bets (but the folder's bet is already in)
        # When someone folds, the other player wins the pot
        # net for winner = loser's bet, net for loser = -loser's bet
        # Actually: each player put in their bet. Winner gets back their bet + opponent's bet.
        # Net for winner = opponent's bet. Net for loser = -their own bet.
        # But bets might differ (the folder might have less in). Let's use min of bets.
        # Actually in the FOLD case, the folder didn't match - let me think again.
        # The fold means the active_team folded. They lose what they put in.
        # The non-folder wins what the folder put in.
        folder = last_active
        if folder == geoz_team:
            # geoz folded, loses their bet
            if geoz_team == 0:
                delta = -last_bet0
            else:
                delta = -last_bet1
        else:
            # opponent folded, geoz wins opponent's bet
            if geoz_team == 0:
                delta = last_bet1
            else:
                delta = last_bet0
        hand_deltas.append((last_h, delta))
    else:
        # Showdown (CHECK at river) - we can't easily determine winner
        # Use a heuristic: assume the pot is bet0 + bet1, and it's roughly 50/50
        # For accuracy, skip this hand (it's 1 out of 1000)
        # Actually, let's try to not count it - the bankroll at start of last hand
        # gives us all prior hands, and this one hand is ~0.1% of total
        # We'll still count it as a hand for hand_count purposes
        # but set delta = 0 (unknown)
        hand_deltas.append((last_h, 0))

    # Compute summary stats
    geoz_net = sum(d for _, d in hand_deltas)
    hands_won = sum(1 for _, d in hand_deltas if d > 0)
    hands_lost = sum(1 for _, d in hand_deltas if d < 0)
    hands_tied = sum(1 for _, d in hand_deltas if d == 0)

    chips_won = sum(d for _, d in hand_deltas if d > 0)
    chips_lost = sum(d for _, d in hand_deltas if d < 0)

    return {
        'team0': team0_name,
        'team1': team1_name,
        'geoz_team': geoz_team,
        'opponent': opp_name,
        'total_hands': total_hands,
        'hand_deltas': hand_deltas,
        'geoz_net': geoz_net,
        'hands_won': hands_won,
        'hands_lost': hands_lost,
        'hands_tied': hands_tied,
        'chips_won': chips_won,
        'chips_lost': chips_lost,
        'match_won': 1 if geoz_net > 0 else (0 if geoz_net < 0 else 0),
    }


# === STEP 3: Aggregate by bot version ===
def main():
    bot_map = build_bot_map()

    # Parse all CSVs
    csv_files = glob.glob(os.path.join(LOG_DIR, '*.csv'))

    # Map CSV to match_id
    all_matches = []
    unmatched = 0
    for cf in csv_files:
        base = os.path.basename(cf)
        m = re.match(r'match_(\d+)', base)
        if not m:
            continue
        match_id = int(m.group(1))

        # Determine bot version
        bot_name = bot_map.get(match_id, None)
        if bot_name is None:
            unmatched += 1
            continue

        result = parse_csv(cf)
        if result is None:
            continue

        result['match_id'] = match_id
        result['bot_name'] = bot_name
        result['bot_family'] = bot_family(bot_name)
        all_matches.append(result)

    print(f"{'='*100}")
    print(f"TOURNAMENT ANALYSIS - {len(all_matches)} matches analyzed ({unmatched} CSVs had no matching .log file)")
    print(f"{'='*100}")

    # Count by bot version
    by_version = defaultdict(list)
    for m in all_matches:
        by_version[m['bot_name']].append(m)

    by_family = defaultdict(list)
    for m in all_matches:
        by_family[m['bot_family']].append(m)

    # ================================================================
    # SECTION 1: Overall by bot version
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 1: OVERALL RESULTS BY BOT VERSION")
    print(f"{'='*100}")

    print(f"\n{'Bot Version':<22} {'Matches':>7} {'W':>4} {'L':>4} {'D':>3} {'Win%':>7} {'Net Chips':>11} {'EV/Hand':>9} {'Hands':>7} {'Hand W%':>8} {'AvgW':>7} {'AvgL':>7}")
    print("-" * 112)

    # Sort by number of matches descending
    for bot_name in sorted(by_version.keys(), key=lambda x: len(by_version[x]), reverse=True):
        matches = by_version[bot_name]
        wins = sum(1 for m in matches if m['match_won'] == 1)
        losses = sum(1 for m in matches if m['geoz_net'] < 0)
        draws = sum(1 for m in matches if m['geoz_net'] == 0)
        total = len(matches)
        win_rate = wins / total * 100 if total > 0 else 0
        net_chips = sum(m['geoz_net'] for m in matches)
        total_hands = sum(m['total_hands'] for m in matches)
        ev_per_hand = net_chips / total_hands if total_hands > 0 else 0

        total_hands_won = sum(m['hands_won'] for m in matches)
        total_hands_played = sum(m['total_hands'] for m in matches)
        hand_win_rate = total_hands_won / total_hands_played * 100 if total_hands_played > 0 else 0

        total_chips_won = sum(m['chips_won'] for m in matches)
        total_chips_lost = sum(m['chips_lost'] for m in matches)
        total_hw = sum(m['hands_won'] for m in matches)
        total_hl = sum(m['hands_lost'] for m in matches)
        avg_win = total_chips_won / total_hw if total_hw > 0 else 0
        avg_loss = total_chips_lost / total_hl if total_hl > 0 else 0

        family_tag = " [PA]" if bot_family(bot_name) == 'PlayerAgent' else " [C2]"
        print(f"{bot_name + family_tag:<22} {total:>7} {wins:>4} {losses:>4} {draws:>3} {win_rate:>6.1f}% {net_chips:>+11} {ev_per_hand:>+9.2f} {total_hands:>7} {hand_win_rate:>7.1f}% {avg_win:>7.1f} {avg_loss:>7.1f}")

    # ================================================================
    # SECTION 2: Overall by bot FAMILY
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 2: OVERALL RESULTS BY BOT FAMILY")
    print(f"{'='*100}")

    print(f"\n{'Bot Family':<18} {'Matches':>7} {'W':>4} {'L':>4} {'D':>3} {'Win%':>7} {'Net Chips':>11} {'EV/Hand':>9} {'Hands':>7} {'Hand W%':>8} {'AvgW':>7} {'AvgL':>7}")
    print("-" * 104)

    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = by_family[family]
        wins = sum(1 for m in matches if m['match_won'] == 1)
        losses = sum(1 for m in matches if m['geoz_net'] < 0)
        draws = sum(1 for m in matches if m['geoz_net'] == 0)
        total = len(matches)
        win_rate = wins / total * 100 if total > 0 else 0
        net_chips = sum(m['geoz_net'] for m in matches)
        total_hands = sum(m['total_hands'] for m in matches)
        ev_per_hand = net_chips / total_hands if total_hands > 0 else 0

        total_hands_won = sum(m['hands_won'] for m in matches)
        total_hands_played = sum(m['total_hands'] for m in matches)
        hand_win_rate = total_hands_won / total_hands_played * 100 if total_hands_played > 0 else 0

        total_chips_won = sum(m['chips_won'] for m in matches)
        total_chips_lost = sum(m['chips_lost'] for m in matches)
        total_hw = sum(m['hands_won'] for m in matches)
        total_hl = sum(m['hands_lost'] for m in matches)
        avg_win = total_chips_won / total_hw if total_hw > 0 else 0
        avg_loss = total_chips_lost / total_hl if total_hl > 0 else 0

        print(f"{family:<18} {total:>7} {wins:>4} {losses:>4} {draws:>3} {win_rate:>6.1f}% {net_chips:>+11} {ev_per_hand:>+9.2f} {total_hands:>7} {hand_win_rate:>7.1f}% {avg_win:>7.1f} {avg_loss:>7.1f}")

    # ================================================================
    # SECTION 3: Per-opponent breakdown for each bot family
    # ================================================================
    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = by_family[family]

        print(f"\n{'='*100}")
        print(f"SECTION 3: PER-OPPONENT BREAKDOWN FOR {family}")
        print(f"{'='*100}")

        # Group by opponent
        by_opp = defaultdict(list)
        for m in matches:
            by_opp[m['opponent']].append(m)

        print(f"\n{'Opponent':<25} {'Matches':>7} {'W':>4} {'L':>4} {'Win%':>7} {'Net Chips':>11} {'EV/Hand':>9} {'Hands':>7} {'Hand W%':>8} {'AvgW':>7} {'AvgL':>7}")
        print("-" * 112)

        # Sort by net chips
        for opp in sorted(by_opp.keys(), key=lambda x: sum(m['geoz_net'] for m in by_opp[x]), reverse=True):
            opp_matches = by_opp[opp]
            wins = sum(1 for m in opp_matches if m['match_won'] == 1)
            losses = sum(1 for m in opp_matches if m['geoz_net'] < 0)
            total = len(opp_matches)
            win_rate = wins / total * 100 if total > 0 else 0
            net_chips = sum(m['geoz_net'] for m in opp_matches)
            total_hands = sum(m['total_hands'] for m in opp_matches)
            ev_per_hand = net_chips / total_hands if total_hands > 0 else 0

            total_hands_won = sum(m['hands_won'] for m in opp_matches)
            hand_win_rate = total_hands_won / total_hands * 100 if total_hands > 0 else 0

            total_chips_won = sum(m['chips_won'] for m in opp_matches)
            total_chips_lost = sum(m['chips_lost'] for m in opp_matches)
            total_hw = sum(m['hands_won'] for m in opp_matches)
            total_hl = sum(m['hands_lost'] for m in opp_matches)
            avg_win = total_chips_won / total_hw if total_hw > 0 else 0
            avg_loss = total_chips_lost / total_hl if total_hl > 0 else 0

            marker = " ***" if net_chips < 0 else ""
            print(f"{opp:<25} {total:>7} {wins:>4} {losses:>4} {win_rate:>6.1f}% {net_chips:>+11} {ev_per_hand:>+9.2f} {total_hands:>7} {hand_win_rate:>7.1f}% {avg_win:>7.1f} {avg_loss:>7.1f}{marker}")

        # Summary of struggling opponents
        struggling = [(opp, sum(m['geoz_net'] for m in by_opp[opp])) for opp in by_opp
                       if sum(m['geoz_net'] for m in by_opp[opp]) < 0]
        struggling.sort(key=lambda x: x[1])

        if struggling:
            print(f"\n  STRUGGLING AGAINST (negative net chips, marked *** above):")
            for opp, net in struggling:
                opp_matches = by_opp[opp]
                w = sum(1 for m in opp_matches if m['match_won'] == 1)
                l = sum(1 for m in opp_matches if m['geoz_net'] < 0)
                print(f"    {opp:<25} net={net:>+8}  W-L={w}-{l}")

    # ================================================================
    # SECTION 4: Per-opponent breakdown for each bot VERSION
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 4: PER-OPPONENT BREAKDOWN BY BOT VERSION (top versions only)")
    print(f"{'='*100}")

    # Only show versions with >= 5 matches
    for bot_name in sorted(by_version.keys(), key=lambda x: len(by_version[x]), reverse=True):
        matches = by_version[bot_name]
        if len(matches) < 5:
            continue

        family_tag = " [PA]" if bot_family(bot_name) == 'PlayerAgent' else " [C2]"
        print(f"\n--- {bot_name}{family_tag} ({len(matches)} matches) ---")

        by_opp = defaultdict(list)
        for m in matches:
            by_opp[m['opponent']].append(m)

        print(f"  {'Opponent':<25} {'M':>3} {'W':>3} {'L':>3} {'Win%':>6} {'Net':>9} {'EV/H':>8}")
        print(f"  {'-'*60}")

        for opp in sorted(by_opp.keys(), key=lambda x: sum(m['geoz_net'] for m in by_opp[x]), reverse=True):
            opp_matches = by_opp[opp]
            w = sum(1 for m in opp_matches if m['match_won'] == 1)
            l = sum(1 for m in opp_matches if m['geoz_net'] < 0)
            total = len(opp_matches)
            wr = w / total * 100 if total > 0 else 0
            net = sum(m['geoz_net'] for m in opp_matches)
            th = sum(m['total_hands'] for m in opp_matches)
            evh = net / th if th > 0 else 0
            marker = " ***" if net < 0 else ""
            print(f"  {opp:<25} {total:>3} {w:>3} {l:>3} {wr:>5.0f}% {net:>+9} {evh:>+8.2f}{marker}")

    # ================================================================
    # SECTION 5: Head-to-head comparison - shared opponents
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 5: HEAD-TO-HEAD COMPARISON - PlayerAgent vs Claude2Agent (by family)")
    print(f"{'='*100}")

    pa_matches = by_family.get('PlayerAgent', [])
    c2_matches = by_family.get('Claude2Agent', [])

    pa_by_opp = defaultdict(list)
    for m in pa_matches:
        pa_by_opp[m['opponent']].append(m)

    c2_by_opp = defaultdict(list)
    for m in c2_matches:
        c2_by_opp[m['opponent']].append(m)

    shared_opps = sorted(set(pa_by_opp.keys()) & set(c2_by_opp.keys()))

    print(f"\n{'Opponent':<22} {'':>3} {'--- PlayerAgent ---':>35} {'':>3} {'--- Claude2Agent ---':>35} {'Better':>10}")
    print(f"{'':.<22} {'M':>3} {'W-L':>6} {'Win%':>6} {'Net':>9} {'EV/H':>8} {'M':>4} {'W-L':>6} {'Win%':>6} {'Net':>9} {'EV/H':>8} {'':>10}")
    print("-" * 120)

    for opp in shared_opps:
        pa_om = pa_by_opp[opp]
        c2_om = c2_by_opp[opp]

        pa_w = sum(1 for m in pa_om if m['match_won'] == 1)
        pa_l = sum(1 for m in pa_om if m['geoz_net'] < 0)
        pa_total = len(pa_om)
        pa_wr = pa_w / pa_total * 100 if pa_total > 0 else 0
        pa_net = sum(m['geoz_net'] for m in pa_om)
        pa_th = sum(m['total_hands'] for m in pa_om)
        pa_evh = pa_net / pa_th if pa_th > 0 else 0

        c2_w = sum(1 for m in c2_om if m['match_won'] == 1)
        c2_l = sum(1 for m in c2_om if m['geoz_net'] < 0)
        c2_total = len(c2_om)
        c2_wr = c2_w / c2_total * 100 if c2_total > 0 else 0
        c2_net = sum(m['geoz_net'] for m in c2_om)
        c2_th = sum(m['total_hands'] for m in c2_om)
        c2_evh = c2_net / c2_th if c2_th > 0 else 0

        if pa_evh > c2_evh:
            better = "PA"
        elif c2_evh > pa_evh:
            better = "C2"
        else:
            better = "TIE"

        print(f"{opp:<22} {pa_total:>3} {pa_w:>2}-{pa_l:<3} {pa_wr:>5.0f}% {pa_net:>+9} {pa_evh:>+8.2f} {c2_total:>4} {c2_w:>2}-{c2_l:<3} {c2_wr:>5.0f}% {c2_net:>+9} {c2_evh:>+8.2f} {better:>10}")

    # Also show opponents unique to each family
    pa_only = sorted(set(pa_by_opp.keys()) - set(c2_by_opp.keys()))
    c2_only = sorted(set(c2_by_opp.keys()) - set(pa_by_opp.keys()))

    if pa_only:
        print(f"\nOpponents only faced by PlayerAgent:")
        for opp in pa_only:
            om = pa_by_opp[opp]
            w = sum(1 for m in om if m['match_won'] == 1)
            l = sum(1 for m in om if m['geoz_net'] < 0)
            net = sum(m['geoz_net'] for m in om)
            th = sum(m['total_hands'] for m in om)
            evh = net / th if th > 0 else 0
            print(f"  {opp:<25} M={len(om):>3}  W-L={w}-{l}  Net={net:>+9}  EV/H={evh:>+.2f}")

    if c2_only:
        print(f"\nOpponents only faced by Claude2Agent:")
        for opp in c2_only:
            om = c2_by_opp[opp]
            w = sum(1 for m in om if m['match_won'] == 1)
            l = sum(1 for m in om if m['geoz_net'] < 0)
            net = sum(m['geoz_net'] for m in om)
            th = sum(m['total_hands'] for m in om)
            evh = net / th if th > 0 else 0
            print(f"  {opp:<25} M={len(om):>3}  W-L={w}-{l}  Net={net:>+9}  EV/H={evh:>+.2f}")

    # ================================================================
    # SECTION 6: Sizing asymmetry analysis
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 6: SIZING ASYMMETRY - Average chips won per win vs lost per loss")
    print(f"{'='*100}")

    print(f"\n{'Bot Version':<22} {'Hands Won':>10} {'Avg Chips/Win':>14} {'Hands Lost':>11} {'Avg Chips/Loss':>15} {'Win:Loss Ratio':>15} {'Edge':>7}")
    print("-" * 100)

    for bot_name in sorted(by_version.keys(), key=lambda x: len(by_version[x]), reverse=True):
        matches = by_version[bot_name]
        total_chips_won = sum(m['chips_won'] for m in matches)
        total_chips_lost = sum(m['chips_lost'] for m in matches)
        total_hw = sum(m['hands_won'] for m in matches)
        total_hl = sum(m['hands_lost'] for m in matches)
        avg_win = total_chips_won / total_hw if total_hw > 0 else 0
        avg_loss = abs(total_chips_lost / total_hl) if total_hl > 0 else 0
        ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        edge = "GOOD" if ratio > 1.0 else "BAD"

        family_tag = " [PA]" if bot_family(bot_name) == 'PlayerAgent' else " [C2]"
        print(f"{bot_name + family_tag:<22} {total_hw:>10} {avg_win:>14.2f} {total_hl:>11} {avg_loss:>15.2f} {ratio:>15.3f} {edge:>7}")

    print(f"\n--- By Family ---")
    print(f"{'Bot Family':<18} {'Hands Won':>10} {'Avg Chips/Win':>14} {'Hands Lost':>11} {'Avg Chips/Loss':>15} {'Win:Loss Ratio':>15} {'Edge':>7}")
    print("-" * 96)

    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = by_family[family]
        total_chips_won = sum(m['chips_won'] for m in matches)
        total_chips_lost = sum(m['chips_lost'] for m in matches)
        total_hw = sum(m['hands_won'] for m in matches)
        total_hl = sum(m['hands_lost'] for m in matches)
        avg_win = total_chips_won / total_hw if total_hw > 0 else 0
        avg_loss = abs(total_chips_lost / total_hl) if total_hl > 0 else 0
        ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        edge = "GOOD" if ratio > 1.0 else "BAD"
        print(f"{family:<18} {total_hw:>10} {avg_win:>14.2f} {total_hl:>11} {avg_loss:>15.2f} {ratio:>15.3f} {edge:>7}")

    # ================================================================
    # SECTION 7: Per-hand win rate analysis
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 7: PER-HAND WIN RATE ANALYSIS")
    print(f"{'='*100}")

    print(f"\n{'Bot Version':<22} {'Total Hands':>11} {'Won':>8} {'Lost':>8} {'Tied':>8} {'Hand Win%':>10} {'Hand Loss%':>11}")
    print("-" * 84)

    for bot_name in sorted(by_version.keys(), key=lambda x: len(by_version[x]), reverse=True):
        matches = by_version[bot_name]
        th = sum(m['total_hands'] for m in matches)
        hw = sum(m['hands_won'] for m in matches)
        hl = sum(m['hands_lost'] for m in matches)
        ht = sum(m['hands_tied'] for m in matches)
        hwr = hw / th * 100 if th > 0 else 0
        hlr = hl / th * 100 if th > 0 else 0

        family_tag = " [PA]" if bot_family(bot_name) == 'PlayerAgent' else " [C2]"
        print(f"{bot_name + family_tag:<22} {th:>11} {hw:>8} {hl:>8} {ht:>8} {hwr:>9.2f}% {hlr:>10.2f}%")

    # ================================================================
    # SECTION 8: Match-level stats
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 8: MATCH-LEVEL DISTRIBUTION (chip results per match)")
    print(f"{'='*100}")

    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = by_family[family]
        nets = sorted([m['geoz_net'] for m in matches])

        print(f"\n--- {family} ({len(matches)} matches) ---")
        if nets:
            print(f"  Min: {min(nets):>+8}  Max: {max(nets):>+8}  Median: {nets[len(nets)//2]:>+8}  Mean: {sum(nets)/len(nets):>+8.1f}")

            # Distribution buckets
            buckets = {'< -2000': 0, '-2000 to -1000': 0, '-1000 to -500': 0, '-500 to 0': 0,
                       '0 to 500': 0, '500 to 1000': 0, '1000 to 2000': 0, '> 2000': 0}
            for n in nets:
                if n < -2000: buckets['< -2000'] += 1
                elif n < -1000: buckets['-2000 to -1000'] += 1
                elif n < -500: buckets['-1000 to -500'] += 1
                elif n < 0: buckets['-500 to 0'] += 1
                elif n < 500: buckets['0 to 500'] += 1
                elif n < 1000: buckets['500 to 1000'] += 1
                elif n < 2000: buckets['1000 to 2000'] += 1
                else: buckets['> 2000'] += 1

            for bucket, count in buckets.items():
                bar = '#' * count
                print(f"  {bucket:>18}: {count:>4} {bar}")

    # ================================================================
    # SECTION 9: Evolution over time (by match_id order)
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 9: PERFORMANCE EVOLUTION OVER TIME (rolling 10-match windows)")
    print(f"{'='*100}")

    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = sorted(by_family[family], key=lambda x: x['match_id'])

        if len(matches) < 10:
            print(f"\n--- {family}: too few matches for rolling analysis ---")
            continue

        print(f"\n--- {family} ({len(matches)} matches) ---")
        print(f"  {'Window':<20} {'Bot Version(s)':<25} {'W-L':>6} {'Net':>9} {'EV/H':>8}")
        print(f"  {'-'*72}")

        window_size = 10
        for i in range(0, len(matches) - window_size + 1, window_size):
            window = matches[i:i + window_size]
            w = sum(1 for m in window if m['match_won'] == 1)
            l = sum(1 for m in window if m['geoz_net'] < 0)
            net = sum(m['geoz_net'] for m in window)
            th = sum(m['total_hands'] for m in window)
            evh = net / th if th > 0 else 0
            versions = sorted(set(m['bot_name'] for m in window))
            ver_str = ', '.join(versions)
            if len(ver_str) > 24:
                ver_str = ver_str[:21] + '...'
            id_range = f"#{window[0]['match_id']}-#{window[-1]['match_id']}"
            print(f"  {id_range:<20} {ver_str:<25} {w:>2}-{l:<3} {net:>+9} {evh:>+8.2f}")

    # ================================================================
    # SECTION 10: Best and worst individual matches
    # ================================================================
    print(f"\n{'='*100}")
    print("SECTION 10: BEST AND WORST INDIVIDUAL MATCHES")
    print(f"{'='*100}")

    for family in sorted(by_family.keys(), key=lambda x: len(by_family[x]), reverse=True):
        matches = by_family[family]
        sorted_matches = sorted(matches, key=lambda x: x['geoz_net'])

        print(f"\n--- {family} ---")

        print(f"  WORST 5 matches:")
        for m in sorted_matches[:5]:
            print(f"    Match #{m['match_id']:>5} vs {m['opponent']:<22} Net={m['geoz_net']:>+8}  Bot={m['bot_name']}")

        print(f"  BEST 5 matches:")
        for m in sorted_matches[-5:]:
            print(f"    Match #{m['match_id']:>5} vs {m['opponent']:<22} Net={m['geoz_net']:>+8}  Bot={m['bot_name']}")


if __name__ == '__main__':
    main()
