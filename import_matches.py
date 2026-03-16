#!/usr/bin/env python3
"""
import_matches.py — Import match bundle from fetch_all_matches_bulk.js

Usage:
  python3 import_matches.py match_bundle_2026-03-15T12-00-00.json

Unpacks CSV and bot log files into tournament_logs/, skipping duplicates.
Prints summary stats and per-opponent breakdown.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_logs")
OUR_TEAM = "geoz"


def detect_our_slot(content):
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
    return 0, "us", "opp"


def analyze_csv(content):
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    if not data_lines:
        return {}
    rows = list(csv.DictReader(data_lines))
    if not rows:
        return {}

    our_slot, our_name, opp_name = detect_our_slot(content)
    our_col = f"team_{our_slot}_bankroll"
    last = rows[-1]
    total_hands = int(last.get("hand_number", 0)) + 1
    our_net = float(last.get(our_col, 0))
    ev = our_net / total_hands if total_hands else 0

    hand_end = {}
    for row in rows:
        h = int(row["hand_number"])
        hand_end[h] = float(row[our_col])
    pnls = []
    sh = sorted(hand_end)
    for i, h in enumerate(sh):
        pnls.append(hand_end[h] if i == 0 else hand_end[h] - hand_end[sh[i - 1]])
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "hands": total_hands,
        "net": our_net,
        "ev": ev,
        "wr": len(wins) / len(pnls) * 100 if pnls else 0,
        "opponent": opp_name,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 import_matches.py <match_bundle.json>")
        sys.exit(1)

    bundle_path = sys.argv[1]
    if not os.path.exists(bundle_path):
        print(f"File not found: {bundle_path}")
        sys.exit(1)

    with open(bundle_path) as f:
        bundle = json.load(f)

    matches = bundle.get("matches", [])
    print(f"\nBundle: {len(matches)} matches, fetched {bundle.get('fetchedAt', '?')}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    existing = set()
    for fname in os.listdir(LOGS_DIR):
        m = re.match(r"match_(\d+)", fname)
        if m:
            existing.add(int(m.group(1)))

    imported = 0
    skipped = 0
    stats = []  # (matchId, opponent, bankroll, ev, wr, botName)

    for m in matches:
        mid = m.get("matchId")
        if not mid:
            continue

        # Save CSV
        if m.get("csv"):
            csv_path = os.path.join(LOGS_DIR, f"match_{mid}.csv")
            if mid not in existing:
                with open(csv_path, "w") as f:
                    f.write(m["csv"])
                imported += 1
            else:
                skipped += 1

            # Analyze
            analysis = analyze_csv(m["csv"])
            stats.append({
                "matchId": mid,
                "opponent": m.get("opponent", analysis.get("opponent", "?")),
                "bankroll": m.get("bankroll", analysis.get("net", 0)),
                "ev": analysis.get("ev", 0),
                "wr": analysis.get("wr", 0),
                "botName": m.get("botName", ""),
                "timestamp": m.get("timestamp", ""),
                "matchType": m.get("matchType", ""),
            })

        # Save bot log
        if m.get("botLog"):
            bot_name = m.get("botName", "bot")
            log_path = os.path.join(LOGS_DIR, f"match_{mid}_bot_{bot_name}.log")
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write(m["botLog"])

    print(f"Imported: {imported} new, {skipped} already existed")

    # Summary
    if not stats:
        return

    total_w = sum(1 for s in stats if s["bankroll"] > 0)
    total_l = sum(1 for s in stats if s["bankroll"] <= 0)
    total_net = sum(s["bankroll"] for s in stats)
    total = total_w + total_l

    print(f"\n{'='*62}")
    print(f"  Overall: {total_w}W-{total_l}L ({total_w/total*100:.1f}% WR)")
    print(f"  Net chips: {total_net:+.0f}")
    print(f"{'='*62}")

    # Per-opponent breakdown
    by_opp = defaultdict(lambda: {"w": 0, "l": 0, "net": 0})
    for s in stats:
        opp = s["opponent"]
        if s["bankroll"] > 0:
            by_opp[opp]["w"] += 1
        else:
            by_opp[opp]["l"] += 1
        by_opp[opp]["net"] += s["bankroll"]

    print(f"\n  {'Opponent':<25} {'W':>3}-{'L':<3} {'WR':>6}  {'Net':>8}")
    print(f"  {'─'*50}")
    for opp, d in sorted(by_opp.items(), key=lambda x: x[1]["net"], reverse=True):
        t = d["w"] + d["l"]
        wr = d["w"] / t * 100 if t else 0
        print(f"  {opp:<25} {d['w']:>3}-{d['l']:<3} {wr:>5.1f}%  {d['net']:>+8.0f}")

    # Per-bot breakdown
    by_bot = defaultdict(lambda: {"w": 0, "l": 0, "net": 0})
    for s in stats:
        bot = s["botName"] or "unknown"
        if s["bankroll"] > 0:
            by_bot[bot]["w"] += 1
        else:
            by_bot[bot]["l"] += 1
        by_bot[bot]["net"] += s["bankroll"]

    if len(by_bot) > 1:
        print(f"\n  {'Bot Version':<25} {'W':>3}-{'L':<3} {'WR':>6}  {'Net':>8}")
        print(f"  {'─'*50}")
        for bot, d in sorted(by_bot.items(), key=lambda x: x[1]["net"], reverse=True):
            t = d["w"] + d["l"]
            wr = d["w"] / t * 100 if t else 0
            print(f"  {bot:<25} {d['w']:>3}-{d['l']:<3} {wr:>5.1f}%  {d['net']:>+8.0f}")

    print()


if __name__ == "__main__":
    main()
