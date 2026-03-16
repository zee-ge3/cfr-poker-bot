#!/usr/bin/env python3
"""
Analyze two tournament match losses for the geoz team's AVBv10 bot (Claude2Agent).
Produces detailed breakdown of why the bot lost each match.
"""

import csv
import re
import sys
from collections import defaultdict, Counter

def parse_match(csv_path, log_path):
    """Parse a match CSV and bot log file."""
    # Read header to determine team mapping
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip()

    # Parse "# Team 0: X, Team 1: Y"
    m = re.match(r'# Team 0: (.+), Team 1: (.+)', header_line)
    team0_name = m.group(1).strip()
    team1_name = m.group(2).strip()

    # Determine which team is geoz
    if 'geoz' in team0_name.lower():
        geoz_team = 0
        opp_name = team1_name
    else:
        geoz_team = 1
        opp_name = team0_name

    # Read CSV rows
    rows = []
    with open(csv_path, 'r') as f:
        f.readline()  # skip comment header
        reader = csv.DictReader(f)
        for row in reader:
            row['hand_number'] = int(row['hand_number'])
            row['active_team'] = int(row['active_team'])
            row['team_0_bankroll'] = int(row['team_0_bankroll'])
            row['team_1_bankroll'] = int(row['team_1_bankroll'])
            row['action_amount'] = int(row['action_amount'])
            row['team_0_bet'] = int(row['team_0_bet'])
            row['team_1_bet'] = int(row['team_1_bet'])
            rows.append(row)

    # Parse bot log for WON/LOST lines and SHOWDOWN info
    log_entries = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # WON/LOST lines
            wl_match = re.search(r'(WON|LOST) ([\d.]+) chips \| cum=([+-]?\d+) hands=(\d+)', line)
            if wl_match:
                log_entries.append({
                    'type': 'result',
                    'outcome': wl_match.group(1),
                    'chips': float(wl_match.group(2)),
                    'cumulative': int(wl_match.group(3)),
                    'hand': int(wl_match.group(4)),
                })
            # SHOWDOWN lines
            sd_match = re.search(r'SHOWDOWN us=(\[.*?\]) opp=(\[.*?\]) board=(\[.*?\])', line)
            if sd_match:
                log_entries.append({
                    'type': 'showdown',
                    'us': sd_match.group(1),
                    'opp': sd_match.group(2),
                    'board': sd_match.group(3),
                })

    return {
        'rows': rows,
        'log_entries': log_entries,
        'geoz_team': geoz_team,
        'opp_name': opp_name,
        'team0_name': team0_name,
        'team1_name': team1_name,
    }


def analyze_match(match_data, match_label):
    """Perform full analysis on a parsed match."""
    rows = match_data['rows']
    log_entries = match_data['log_entries']
    geoz_team = match_data['geoz_team']
    opp_name = match_data['opp_name']

    print("=" * 80)
    print(f"  MATCH ANALYSIS: {match_label}")
    print(f"  geoz (Team {geoz_team}) vs {opp_name} (Team {1 - geoz_team})")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. SUMMARY
    # -------------------------------------------------------------------------
    # Get hand results from log
    results = [e for e in log_entries if e['type'] == 'result']
    showdowns = [e for e in log_entries if e['type'] == 'showdown']

    total_hands = len(results)
    wins = sum(1 for r in results if r['outcome'] == 'WON')
    losses = sum(1 for r in results if r['outcome'] == 'LOST')
    final_cum = results[-1]['cumulative'] if results else 0

    print(f"\n--- 1. SUMMARY ---")
    print(f"Opponent:        {opp_name}")
    print(f"Total hands:     {total_hands}")
    print(f"geoz final net:  {final_cum:+d} chips")
    print(f"Hands won:       {wins} ({100*wins/total_hands:.1f}%)")
    print(f"Hands lost:      {losses} ({100*losses/total_hands:.1f}%)")
    print(f"Showdowns seen:  {len(showdowns)}")

    # -------------------------------------------------------------------------
    # 2. BANKROLL CURVE
    # -------------------------------------------------------------------------
    print(f"\n--- 2. BANKROLL CURVE (geoz net at hand milestones) ---")

    # Build hand->cumulative mapping from log
    hand_cum = {}
    for r in results:
        hand_cum[r['hand']] = r['cumulative']

    milestones = list(range(0, 1000, 100)) + [999]
    print(f"  {'Hand':>6s}  {'geoz net':>10s}  {'bar'}")
    for h in milestones:
        # Find closest hand at or before milestone
        closest = None
        for hh in sorted(hand_cum.keys()):
            if hh <= h:
                closest = hh
        if closest is not None:
            val = hand_cum[closest]
            bar_len = val // 5 if val > 0 else -((-val) // 5)
            if bar_len >= 0:
                bar = "+" * min(bar_len, 60)
            else:
                bar = "-" * min(-bar_len, 60)
            print(f"  {h:>6d}  {val:>+10d}  |{bar}")
        else:
            print(f"  {h:>6d}  {'N/A':>10s}")

    # -------------------------------------------------------------------------
    # 3. BIGGEST LOSING HANDS
    # -------------------------------------------------------------------------
    print(f"\n--- 3. TOP 10 BIGGEST LOSING HANDS ---")

    losing_results = [r for r in results if r['outcome'] == 'LOST']
    losing_results.sort(key=lambda r: r['chips'], reverse=True)

    # Group CSV rows by hand
    hands_by_num = defaultdict(list)
    for row in rows:
        hands_by_num[row['hand_number']].append(row)

    top_losers = losing_results[:10]
    for i, lr in enumerate(top_losers):
        hand_num = lr['hand']
        chips_lost = lr['chips']
        hand_rows = hands_by_num[hand_num]

        # Find the last action by geoz (before any DISCARD)
        geoz_actions = [r for r in hand_rows if r['active_team'] == geoz_team
                        and r['action_type'] in ('FOLD', 'CALL', 'RAISE', 'CHECK')]
        last_geoz_action = geoz_actions[-1] if geoz_actions else None

        # Find the last street
        last_street = hand_rows[-1]['street'] if hand_rows else 'N/A'

        # Find if there's a corresponding showdown
        # Look at what action ended the hand
        final_action = hand_rows[-1]['action_type'] if hand_rows else 'N/A'
        ending_action = last_geoz_action['action_type'] if last_geoz_action else 'N/A'

        # Get geoz cards if visible
        geoz_cards_key = f'team_{geoz_team}_cards'
        geoz_cards = hand_rows[0][geoz_cards_key] if hand_rows else 'N/A'

        print(f"  #{i+1}: Hand {hand_num:>4d} | Lost {chips_lost:>6.0f} chips | "
              f"geoz last action: {ending_action:>5s} | ended street: {last_street:>10s} | "
              f"geoz cards: {geoz_cards}")

    # -------------------------------------------------------------------------
    # 4. ACTION ANALYSIS
    # -------------------------------------------------------------------------
    print(f"\n--- 4. ACTION ANALYSIS ---")

    geoz_actions_count = Counter()
    opp_actions_count = Counter()

    for row in rows:
        action = row['action_type']
        if action == 'DISCARD':
            continue
        team = row['active_team']
        if team == geoz_team:
            geoz_actions_count[action] += 1
        else:
            opp_actions_count[action] += 1

    all_actions = sorted(set(list(geoz_actions_count.keys()) + list(opp_actions_count.keys())))

    print(f"  {'Action':>8s}  {'geoz':>8s}  {'%':>6s}  {opp_name:>8s}  {'%':>6s}")
    geoz_total = sum(geoz_actions_count.values())
    opp_total = sum(opp_actions_count.values())
    for act in all_actions:
        gc = geoz_actions_count[act]
        oc = opp_actions_count[act]
        gp = 100 * gc / geoz_total if geoz_total else 0
        op = 100 * oc / opp_total if opp_total else 0
        print(f"  {act:>8s}  {gc:>8d}  {gp:>5.1f}%  {oc:>8d}  {op:>5.1f}%")

    # Aggression factor = (raises + bets) / calls
    geoz_agg = (geoz_actions_count['RAISE']) / max(geoz_actions_count['CALL'], 1)
    opp_agg = (opp_actions_count['RAISE']) / max(opp_actions_count['CALL'], 1)
    print(f"\n  Aggression Factor (RAISE/CALL):")
    print(f"    geoz:  {geoz_agg:.2f}")
    print(f"    {opp_name}:  {opp_agg:.2f}")

    # Fold-to-raise rate
    # Count how many times geoz faced a raise and folded vs called/raised
    geoz_faced_raise = 0
    geoz_folded_to_raise = 0
    opp_faced_raise = 0
    opp_folded_to_raise = 0

    for hand_num, hand_rows in hands_by_num.items():
        for i, row in enumerate(hand_rows):
            if row['action_type'] == 'RAISE':
                raiser = row['active_team']
                # Look for the next non-DISCARD action by the other team
                for j in range(i+1, len(hand_rows)):
                    next_row = hand_rows[j]
                    if next_row['action_type'] == 'DISCARD':
                        continue
                    if next_row['active_team'] != raiser:
                        responder = next_row['active_team']
                        response = next_row['action_type']
                        if responder == geoz_team:
                            geoz_faced_raise += 1
                            if response == 'FOLD':
                                geoz_folded_to_raise += 1
                        else:
                            opp_faced_raise += 1
                            if response == 'FOLD':
                                opp_folded_to_raise += 1
                        break

    geoz_ftr = 100 * geoz_folded_to_raise / max(geoz_faced_raise, 1)
    opp_ftr = 100 * opp_folded_to_raise / max(opp_faced_raise, 1)
    print(f"\n  Fold-to-Raise Rate:")
    print(f"    geoz:  {geoz_folded_to_raise}/{geoz_faced_raise} = {geoz_ftr:.1f}%")
    print(f"    {opp_name}:  {opp_folded_to_raise}/{opp_faced_raise} = {opp_ftr:.1f}%")

    # Pre-flop fold rate (how often fold pre-flop specifically)
    geoz_pf_fold = 0
    opp_pf_fold = 0
    geoz_pf_total = 0
    opp_pf_total = 0
    for row in rows:
        if row['street'] == 'Pre-Flop' and row['action_type'] != 'DISCARD':
            if row['active_team'] == geoz_team:
                geoz_pf_total += 1
                if row['action_type'] == 'FOLD':
                    geoz_pf_fold += 1
            else:
                opp_pf_total += 1
                if row['action_type'] == 'FOLD':
                    opp_pf_fold += 1

    print(f"\n  Pre-Flop Fold Rate:")
    print(f"    geoz:  {geoz_pf_fold}/{geoz_pf_total} = {100*geoz_pf_fold/max(geoz_pf_total,1):.1f}%")
    print(f"    {opp_name}:  {opp_pf_fold}/{opp_pf_total} = {100*opp_pf_fold/max(opp_pf_total,1):.1f}%")

    # -------------------------------------------------------------------------
    # 5. RAISE WAR ANALYSIS
    # -------------------------------------------------------------------------
    print(f"\n--- 5. RAISE WAR ANALYSIS (3+ raises on a single street) ---")

    raise_wars = []
    for hand_num in sorted(hands_by_num.keys()):
        hand_rows = hands_by_num[hand_num]
        # Group by street
        streets = defaultdict(list)
        for row in hand_rows:
            streets[row['street']].append(row)

        for street, street_rows in streets.items():
            raises_in_street = [r for r in street_rows if r['action_type'] == 'RAISE']
            if len(raises_in_street) >= 3:
                # Find max bet on this street
                max_bet = max(r['team_0_bet'] for r in street_rows)
                max_bet2 = max(r['team_1_bet'] for r in street_rows)
                pot_at_stake = max(max_bet, max_bet2)

                # Who won this hand?
                hand_result = [r for r in results if r['hand'] == hand_num]
                if hand_result:
                    outcome = hand_result[0]['outcome']
                    chips = hand_result[0]['chips']
                else:
                    outcome = '?'
                    chips = 0

                # Count raises by each team
                geoz_raises = sum(1 for r in raises_in_street if r['active_team'] == geoz_team)
                opp_raises = sum(1 for r in raises_in_street if r['active_team'] != geoz_team)

                raise_wars.append({
                    'hand': hand_num,
                    'street': street,
                    'num_raises': len(raises_in_street),
                    'pot_at_stake': pot_at_stake,
                    'geoz_raises': geoz_raises,
                    'opp_raises': opp_raises,
                    'outcome': outcome,
                    'chips': chips,
                })

    if raise_wars:
        geoz_rw_wins = sum(1 for rw in raise_wars if rw['outcome'] == 'WON')
        geoz_rw_losses = sum(1 for rw in raise_wars if rw['outcome'] == 'LOST')
        total_rw_chips_won = sum(rw['chips'] for rw in raise_wars if rw['outcome'] == 'WON')
        total_rw_chips_lost = sum(rw['chips'] for rw in raise_wars if rw['outcome'] == 'LOST')

        print(f"  Total raise wars: {len(raise_wars)}")
        print(f"  geoz won: {geoz_rw_wins}, lost: {geoz_rw_losses}")
        print(f"  Chips won in RWs: {total_rw_chips_won:.0f}, lost: {total_rw_chips_lost:.0f}, net: {total_rw_chips_won - total_rw_chips_lost:+.0f}")
        print()
        print(f"  {'Hand':>6s}  {'Street':>10s}  {'#Raises':>7s}  {'Pot':>6s}  {'geoz_R':>6s}  {'opp_R':>6s}  {'Result':>8s}  {'Chips':>6s}")
        for rw in raise_wars:
            print(f"  {rw['hand']:>6d}  {rw['street']:>10s}  {rw['num_raises']:>7d}  {rw['pot_at_stake']:>6d}  "
                  f"{rw['geoz_raises']:>6d}  {rw['opp_raises']:>6d}  {rw['outcome']:>8s}  {rw['chips']:>6.0f}")
    else:
        print("  No raise wars found.")

    # -------------------------------------------------------------------------
    # 6. BLOWOUT HANDS (lost 100 chips)
    # -------------------------------------------------------------------------
    print(f"\n--- 6. BLOWOUT HANDS (geoz lost 100 chips) ---")

    blowouts = [r for r in results if r['outcome'] == 'LOST' and r['chips'] >= 99]
    print(f"  Total blowout losses: {len(blowouts)}")

    # Also count opponent's 100-chip losses (geoz 100-chip wins)
    big_wins = [r for r in results if r['outcome'] == 'WON' and r['chips'] >= 99]
    print(f"  Total blowout wins:   {len(big_wins)}")
    print(f"  Net from blowouts:    {len(big_wins)*100 - len(blowouts)*100:+d} chips")
    print()

    for bo in blowouts:
        hand_num = bo['hand']
        hand_rows = hands_by_num[hand_num]

        geoz_cards_key = f'team_{geoz_team}_cards'
        opp_cards_key = f'team_{1-geoz_team}_cards'

        # Get cards (from first row that has them)
        geoz_cards = hand_rows[0][geoz_cards_key] if hand_rows else 'N/A'
        opp_cards = hand_rows[0][opp_cards_key] if hand_rows else 'N/A'
        board = hand_rows[-1]['board_cards'] if hand_rows else 'N/A'

        # What street did the all-in happen?
        # Find where bets reached 100
        allin_street = 'N/A'
        for row in hand_rows:
            if row[f'team_{geoz_team}_bet'] >= 100 or row[f'team_{1-geoz_team}_bet'] >= 100:
                allin_street = row['street']
                break

        # Get the action sequence for this hand
        action_seq = []
        for row in hand_rows:
            if row['action_type'] != 'DISCARD':
                who = 'G' if row['active_team'] == geoz_team else 'O'
                action_seq.append(f"{who}:{row['action_type'][:1]}{row['street'][0]}")

        print(f"  Hand {hand_num:>4d} | All-in street: {allin_street:>10s} | "
              f"geoz: {geoz_cards} | board: {board}")
        print(f"           Actions: {' '.join(action_seq[:20])}")
        print()

    # -------------------------------------------------------------------------
    # 7. PATTERNS
    # -------------------------------------------------------------------------
    print(f"\n--- 7. PATTERN ANALYSIS ---")

    # a) Chip bleeding: count small losses (fold to raise)
    small_folds = [r for r in results if r['outcome'] == 'LOST' and r['chips'] <= 4]
    medium_losses = [r for r in results if r['outcome'] == 'LOST' and 4 < r['chips'] <= 20]
    large_losses = [r for r in results if r['outcome'] == 'LOST' and r['chips'] > 20]

    small_wins = [r for r in results if r['outcome'] == 'WON' and r['chips'] <= 4]
    medium_wins = [r for r in results if r['outcome'] == 'WON' and 4 < r['chips'] <= 20]
    large_wins = [r for r in results if r['outcome'] == 'WON' and r['chips'] > 20]

    print(f"\n  Loss distribution:")
    print(f"    Small (1-4 chips):    {len(small_folds):>4d} hands, total {sum(r['chips'] for r in small_folds):>6.0f} chips lost")
    print(f"    Medium (5-20 chips):  {len(medium_losses):>4d} hands, total {sum(r['chips'] for r in medium_losses):>6.0f} chips lost")
    print(f"    Large (>20 chips):    {len(large_losses):>4d} hands, total {sum(r['chips'] for r in large_losses):>6.0f} chips lost")

    print(f"\n  Win distribution:")
    print(f"    Small (1-4 chips):    {len(small_wins):>4d} hands, total {sum(r['chips'] for r in small_wins):>6.0f} chips won")
    print(f"    Medium (5-20 chips):  {len(medium_wins):>4d} hands, total {sum(r['chips'] for r in medium_wins):>6.0f} chips won")
    print(f"    Large (>20 chips):    {len(large_wins):>4d} hands, total {sum(r['chips'] for r in large_wins):>6.0f} chips won")

    total_lost = sum(r['chips'] for r in results if r['outcome'] == 'LOST')
    total_won = sum(r['chips'] for r in results if r['outcome'] == 'WON')
    print(f"\n  Total chips lost:  {total_lost:>8.0f}")
    print(f"  Total chips won:   {total_won:>8.0f}")
    print(f"  Net:               {total_won - total_lost:>+8.0f}")

    # b) Street-level fold analysis: where does geoz fold most?
    print(f"\n  Where geoz folds:")
    fold_by_street = Counter()
    for row in rows:
        if row['active_team'] == geoz_team and row['action_type'] == 'FOLD':
            fold_by_street[row['street']] += 1
    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        ct = fold_by_street.get(street, 0)
        print(f"    {street:>10s}: {ct:>4d}")

    # c) Where opponent folds
    print(f"\n  Where {opp_name} folds:")
    opp_fold_by_street = Counter()
    for row in rows:
        if row['active_team'] != geoz_team and row['action_type'] == 'FOLD':
            opp_fold_by_street[row['street']] += 1
    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        ct = opp_fold_by_street.get(street, 0)
        print(f"    {street:>10s}: {ct:>4d}")

    # d) Opponent's raise patterns
    print(f"\n  Opponent raise sizes (post-flop):")
    opp_raise_sizes = []
    for row in rows:
        if row['active_team'] != geoz_team and row['action_type'] == 'RAISE' and row['street'] != 'Pre-Flop':
            opp_raise_sizes.append(row['action_amount'])
    if opp_raise_sizes:
        opp_raise_sizes.sort()
        print(f"    Count: {len(opp_raise_sizes)}")
        print(f"    Min: {min(opp_raise_sizes)}, Max: {max(opp_raise_sizes)}, Median: {opp_raise_sizes[len(opp_raise_sizes)//2]}")
        print(f"    Mean: {sum(opp_raise_sizes)/len(opp_raise_sizes):.1f}")

    # e) How often does geoz fold to a post-flop raise?
    geoz_fold_to_pf_raise = 0
    geoz_face_pf_raise = 0
    for hand_num, hand_rows in hands_by_num.items():
        for i, row in enumerate(hand_rows):
            if (row['active_team'] != geoz_team and row['action_type'] == 'RAISE'
                and row['street'] != 'Pre-Flop'):
                # Find geoz response
                for j in range(i+1, len(hand_rows)):
                    nrow = hand_rows[j]
                    if nrow['action_type'] == 'DISCARD':
                        continue
                    if nrow['active_team'] == geoz_team:
                        geoz_face_pf_raise += 1
                        if nrow['action_type'] == 'FOLD':
                            geoz_fold_to_pf_raise += 1
                        break

    print(f"\n  geoz fold to POST-FLOP raise: {geoz_fold_to_pf_raise}/{geoz_face_pf_raise} = "
          f"{100*geoz_fold_to_pf_raise/max(geoz_face_pf_raise,1):.1f}%")

    # f) How often does opponent raise after geoz checks?
    check_raise_count = 0
    check_count = 0
    for hand_num, hand_rows in hands_by_num.items():
        for i, row in enumerate(hand_rows):
            if row['active_team'] == geoz_team and row['action_type'] == 'CHECK':
                check_count += 1
                # Next action by opponent on same street
                for j in range(i+1, len(hand_rows)):
                    nrow = hand_rows[j]
                    if nrow['street'] != row['street']:
                        break
                    if nrow['action_type'] == 'DISCARD':
                        continue
                    if nrow['active_team'] != geoz_team:
                        if nrow['action_type'] == 'RAISE':
                            check_raise_count += 1
                        break

    print(f"\n  Opponent raises after geoz checks: {check_raise_count}/{check_count} = "
          f"{100*check_raise_count/max(check_count,1):.1f}%")

    # g) Average chips lost per hand by street (how deep does geoz go before losing?)
    print(f"\n  geoz losses by ending street:")
    loss_by_end_street = defaultdict(list)
    for r in results:
        if r['outcome'] == 'LOST':
            hand_num = r['hand']
            hand_rows = hands_by_num[hand_num]
            last_street = hand_rows[-1]['street'] if hand_rows else 'Unknown'
            loss_by_end_street[last_street].append(r['chips'])

    for street in ['Pre-Flop', 'Flop', 'Turn', 'River']:
        if street in loss_by_end_street:
            losses_list = loss_by_end_street[street]
            total = sum(losses_list)
            avg = total / len(losses_list)
            print(f"    {street:>10s}: {len(losses_list):>4d} hands, total {total:>6.0f} chips, avg {avg:.1f} chips/hand")

    # h) Opponent open-raise frequency (how often does opponent raise when first to act preflop?)
    opp_open_raise = 0
    opp_first_act = 0
    for hand_num, hand_rows in hands_by_num.items():
        # First non-discard action
        for row in hand_rows:
            if row['action_type'] != 'DISCARD' and row['street'] == 'Pre-Flop':
                if row['active_team'] != geoz_team:
                    opp_first_act += 1
                    if row['action_type'] == 'RAISE':
                        opp_open_raise += 1
                break

    print(f"\n  Opponent open-raise rate: {opp_open_raise}/{opp_first_act} = "
          f"{100*opp_open_raise/max(opp_first_act,1):.1f}%")

    print()
    return {
        'final_net': final_cum,
        'total_hands': total_hands,
        'wins': wins,
        'losses': losses,
        'blowout_losses': len(blowouts),
        'blowout_wins': len(big_wins),
        'raise_wars': raise_wars,
        'geoz_ftr': geoz_ftr,
        'opp_ftr': opp_ftr,
        'geoz_agg': geoz_agg,
        'opp_agg': opp_agg,
    }


def main():
    matches = [
        {
            'csv': '/home/g30rgez/poker/poker-engine-2026/tournament_logs/match_4940_Claude2Agent.csv',
            'log': '/home/g30rgez/poker/poker-engine-2026/tournament_logs/match_4940_bot_Claude2Agent.log',
            'label': 'Match 4940',
        },
        {
            'csv': '/home/g30rgez/poker/poker-engine-2026/tournament_logs/match_4712_Claude2Agent.csv',
            'log': '/home/g30rgez/poker/poker-engine-2026/tournament_logs/match_4712_bot_Claude2Agent.log',
            'label': 'Match 4712',
        },
    ]

    summaries = []
    for m in matches:
        data = parse_match(m['csv'], m['log'])
        summary = analyze_match(data, m['label'])
        summaries.append((m['label'], summary))

    # -------------------------------------------------------------------------
    # CROSS-MATCH COMPARISON
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("  CROSS-MATCH COMPARISON & KEY TAKEAWAYS")
    print("=" * 80)

    for label, s in summaries:
        print(f"\n  {label}:")
        print(f"    Final net: {s['final_net']:+d}, Win rate: {100*s['wins']/s['total_hands']:.1f}%")
        print(f"    Blowout W/L: {s['blowout_wins']}W / {s['blowout_losses']}L = {(s['blowout_wins']-s['blowout_losses'])*100:+d} net")
        print(f"    Aggression: geoz={s['geoz_agg']:.2f}, opp={s['opp_agg']:.2f}")
        print(f"    Fold-to-raise: geoz={s['geoz_ftr']:.1f}%, opp={s['opp_ftr']:.1f}%")
        rw_count = len(s['raise_wars'])
        rw_net = sum(rw['chips'] for rw in s['raise_wars'] if rw['outcome'] == 'WON') - \
                 sum(rw['chips'] for rw in s['raise_wars'] if rw['outcome'] == 'LOST')
        print(f"    Raise wars: {rw_count}, net {rw_net:+.0f} chips")

    print(f"\n  --- DIAGNOSIS ---")
    print(f"  Key patterns to investigate:")
    print(f"  1. FOLD-TO-RAISE EXPLOITATION: If geoz folds too often to raises,")
    print(f"     opponents can profitably bluff-raise any time geoz checks.")
    print(f"  2. BLOWOUT IMBALANCE: Net from 100-chip pots determines the match.")
    print(f"     Losing even 1-2 more blowouts than winning is devastating.")
    print(f"  3. SMALL POT BLEEDING: Folding to min-raises pre-flop loses 2 chips/hand.")
    print(f"     Over 1000 hands, that's 200+ chips of pure bleed.")
    print(f"  4. CHECK-RAISE VULNERABILITY: If opponent raises every time geoz checks,")
    print(f"     geoz is giving up equity by being passive post-flop.")
    print()


if __name__ == '__main__':
    main()
