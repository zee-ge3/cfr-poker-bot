#!/usr/bin/env python3
"""
CMU AI Poker Tournament — leaderboard poller + log watcher.

Runs two things in parallel:
  1. Polls https://aipoker.cmudsc.com/leaderboard every 60s.
     Prints a compact standings table. Alerts when geoz's match count
     increases (new match played) or ELO changes significantly.

  2. Watches watch_urls.txt for new presigned S3 URLs.
     Auto-downloads and analyzes any new match CSV.

Usage:
  python poll_leaderboard.py

Add new match URLs to watch_urls.txt as you receive them.
"""
import csv
import json
import os
import re
import sys
import time
import threading
import urllib.request
from io import StringIO
from datetime import datetime

LEADERBOARD_URL = "https://aipoker.cmudsc.com/leaderboard"
OUR_TEAM        = "geoz"
WATCH_FILE      = os.path.join(os.path.dirname(__file__), "watch_urls.txt")
LOGS_DIR        = os.path.join(os.path.dirname(__file__), "tournament_logs")
POLL_INTERVAL   = 60   # seconds between leaderboard checks


# ── Leaderboard parsing ──────────────────────────────────────────────────────

def fetch_leaderboard():
    """Fetch and parse team data from the leaderboard page."""
    req = urllib.request.Request(
        LEADERBOARD_URL,
        headers={"User-Agent": "Mozilla/5.0 (poker-bot-monitor/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8")
    return parse_teams(html)


def html_unescape(s: str) -> str:
    return (s.replace("&amp;", "&").replace("&#x27;", "'")
             .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))


def parse_teams(html: str) -> list[dict]:
    """Extract team rows from the server-rendered HTML table."""
    teams = []
    for row in re.split(r'<tr class="border-b', html)[1:]:
        rank_m  = re.search(r'#<!-- -->(\d+)', row)
        name_m  = re.search(r'<span class="transition-colors duration-200">([^<]+)</span>', row)
        nums    = re.findall(r'text-right">(\d+)</td>', row)
        wr_m    = re.search(r'<span>([\d.]+)<!-- -->%</span>', row)
        elo_ch_m = re.search(r'text-(?:red|green)-\d+">([+-]?\d+)</span>', row)

        if not (rank_m and name_m and wr_m and len(nums) >= 3):
            continue

        teams.append({
            "rank":       int(rank_m.group(1)),
            "name":       html_unescape(name_m.group(1)),
            "elo":        int(nums[1]),
            "elo_change": int(elo_ch_m.group(1)) if elo_ch_m else 0,
            "winrate":    float(wr_m.group(1)),
            "matches":    int(nums[2]),
        })
    return teams


def print_standings(teams: list[dict], prev: dict | None):
    """Print compact standings, highlighting geoz and changes."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─'*60}")
    print(f"  Leaderboard  {now}")
    print(f"{'─'*60}")
    print(f"  {'#':<4} {'Team':<22} {'ELO':>5} {'Δ3h':>5}  {'Win%':>6}  {'M':>4}")
    print(f"  {'─'*52}")

    our_data = None
    for t in teams:
        name = t["name"][:21]
        elo  = t["elo"]
        ch   = t.get("elo_change", 0)
        wr   = t["winrate"]
        mc   = t["matches"]
        ch_str = f"{ch:+d}" if ch else "    "
        new_m = ""
        if prev and t["name"] in prev:
            delta_mc = mc - prev[t["name"]]["matches"]
            if delta_mc > 0:
                new_m = f" +{delta_mc}M"

        marker = " ◄" if t["name"] == OUR_TEAM else ""
        print(f"  {t['rank']:<4} {name:<22} {elo:>5} {ch_str:>5}  {wr:>5.1f}%  {mc:>4}{new_m}{marker}")
        if t["name"] == OUR_TEAM:
            our_data = t

    print(f"{'─'*60}")
    return our_data


def leaderboard_loop(state: dict):
    prev_teams = None
    while state["running"]:
        try:
            teams = fetch_leaderboard()
            if teams:
                prev_map = {t["name"]: t for t in prev_teams} if prev_teams else None
                our = print_standings(teams, prev_map)

                # Alert on new matches
                if prev_map and our and our["name"] in prev_map:
                    old = prev_map[our["name"]]
                    if our["matches"] > old["matches"]:
                        n = our["matches"] - old["matches"]
                        print(f"\n  ⚡ {n} new match(es) for {OUR_TEAM}! "
                              f"ELO {old['elo']} → {our['elo']}")

                prev_teams = teams
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Could not parse leaderboard")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Leaderboard error: {e}")

        for _ in range(POLL_INTERVAL):
            if not state["running"]:
                break
            time.sleep(1)


# ── Match log analysis ───────────────────────────────────────────────────────

def download_csv(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return None


def detect_our_slot(content: str) -> tuple[int, str, str]:
    """Return (our_slot, our_name, opp_name) from the header comment."""
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


def parse_watch_file(path: str) -> list[tuple[str, str]]:
    """Parse watch_urls.txt, returning (url, bot_name) pairs.
    Respects '# bot: BotName' directives."""
    entries = []
    current_bot = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"#\s*bot:\s*(.+)", line, re.IGNORECASE)
            if m:
                current_bot = m.group(1).strip()
                continue
            if line.startswith("#"):
                continue
            entries.append((line, current_bot))
    return entries


def analyze_match(content: str, match_id: str, bot_name: str = ""):
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    if not data_lines:
        return
    rows = list(csv.DictReader(data_lines))
    if not rows:
        return

    our_slot, our_name, opp_name = detect_our_slot(content)
    our_col = f"team_{our_slot}_bankroll"
    opp_col = f"team_{1 - our_slot}_bankroll"

    last = rows[-1]
    total_hands = int(last.get("hand_number", 0)) + 1
    our_net = float(last.get(our_col, 0))
    opp_net = float(last.get(opp_col, 0))
    ev = our_net / total_hands if total_hands else 0

    # Per-hand P&L
    hand_end = {}
    for row in rows:
        h = int(row["hand_number"])
        hand_end[h] = float(row[our_col])

    pnls = []
    sh = sorted(hand_end)
    for i, h in enumerate(sh):
        pnls.append(hand_end[h] if i == 0 else hand_end[h] - hand_end[sh[i-1]])

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    result = "WON" if our_net > 0 else "LOST"

    time_info = ""
    if "time_used" in last:
        tu = float(last["time_used"])
        time_info = f"\n  Time used : {tu:.1f}s  ({tu/total_hands*1000:.0f}ms/hand)"

    bot_tag = f"  [{bot_name}]" if bot_name else ""
    print(f"""
{'='*58}
  MATCH {match_id}{bot_tag}  —  {result}
  Hands    : {total_hands}
  {our_name} (us): {our_net:+.0f}   {opp_name}: {opp_net:+.0f}
  EV/hand  : {ev:+.2f}
  Win rate : {len(wins)}/{len(pnls)} = {len(wins)/len(pnls)*100:.1f}%
  Avg W/L  : +{sum(wins)/len(wins):.1f} / {sum(losses)/len(losses):.1f}{time_info}
{'='*58}""")


def analyze_bot_log(content: str, match_id: str, bot_name: str = ""):
    lines = content.splitlines()

    agent_name = bot_name
    if not agent_name:
        name_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (\w+) - \w+ -')
        for l in lines:
            m = name_re.match(l)
            if m:
                agent_name = m.group(1)
                break

    won  = [float(m.group(1)) for l in lines for m in [re.search(r'WON ([\d.]+)', l)] if m]
    lost = [float(m.group(1)) for l in lines for m in [re.search(r'LOST ([\d.]+)', l)] if m]
    total = len(won) + len(lost)
    if not total:
        print("  [WARN] No WON/LOST entries found in bot log")
        return

    net = sum(won) - sum(lost)
    bot_label = f"[{agent_name}]" if agent_name else ""
    print(f"""
{'='*58}
  BOT LOG {match_id}  {bot_label}  ({total} hands)
  Net: {net:+.0f}   EV/h: {net/total:+.2f}
  Win rate: {len(won)}/{total} = {len(won)/total*100:.1f}%
{'='*58}""")


def url_watcher_loop(state: dict):
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(WATCH_FILE):
        open(WATCH_FILE, "w").close()
    print(f"Watching {WATCH_FILE} for new URLs...")

    # Mark a URL seen only if its output file already exists on disk.
    # Clearing tournament_logs/ causes re-download on next start.
    seen = set()
    for url, bot_name in parse_watch_file(WATCH_FILE):
        url_path = url.split("?")[0]
        is_log = url_path.endswith(".log")
        ext = ".log" if is_log else ".csv"
        mid = "?"
        if "match_" in url:
            mid = url.split("match_")[-1].split(".")[0].split("?")[0]
        bot_suffix = f"_{bot_name}" if bot_name else ""
        fname = os.path.join(LOGS_DIR, f"match_{mid}{bot_suffix}{ext}")
        if os.path.exists(fname):
            seen.add(url)

    while state["running"]:
        try:
            for url, bot_name in parse_watch_file(WATCH_FILE):
                if url in seen:
                    continue
                seen.add(url)
                url_path = url.split("?")[0]
                is_log = url_path.endswith(".log")
                ext = ".log" if is_log else ".csv"
                mid = "?"
                if "match_" in url:
                    mid = url.split("match_")[-1].split(".")[0].split("?")[0]
                bot_suffix = f"_{bot_name}" if bot_name else ""
                bot_tag = f" [{bot_name}]" if bot_name else ""
                print(f"\n[NEW] {'Bot log' if is_log else 'Match CSV'} {mid}{bot_tag} — downloading...")
                content = download_csv(url)
                if content:
                    fpath = os.path.join(LOGS_DIR, f"match_{mid}{bot_suffix}{ext}")
                    with open(fpath, "w") as out:
                        out.write(content)
                    print(f"  Saved → {fpath}")
                    if is_log:
                        analyze_bot_log(content, mid, bot_name=bot_name)
                    else:
                        analyze_match(content, mid, bot_name=bot_name)
        except Exception as e:
            print(f"[URL watcher error] {e}")
        time.sleep(2)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    state = {"running": True}
    t1 = threading.Thread(target=leaderboard_loop, args=(state,), daemon=True)
    t2 = threading.Thread(target=url_watcher_loop, args=(state,), daemon=True)
    t1.start()
    t2.start()
    print(f"CMU AI Poker monitor running. Ctrl+C to stop.")
    print(f"Leaderboard polls every {POLL_INTERVAL}s.")
    print(f"Paste match URLs into: {WATCH_FILE}\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        state["running"] = False
        print("\nStopped.")
