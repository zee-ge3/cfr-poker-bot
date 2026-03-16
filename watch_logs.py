#!/usr/bin/env python3
"""
Tournament log watcher.

Usage:
  python watch_logs.py

Monitors watch_urls.txt for new presigned S3 URLs.
When a new URL is added, automatically downloads the CSV and prints analysis.

To add a new match: append the URL to watch_urls.txt (one URL per line).

To tag which bot is active, add a directive before the URLs:
  # bot: PlayerAgent
  https://...url1
  https://...url2
  # bot: Claude2Agent
  https://...url3

The bot tag is saved in the filename and shown in analysis output.
"""
import csv
import os
import sys
import time
import urllib.request
from io import StringIO

WATCH_FILE = os.path.join(os.path.dirname(__file__), "watch_urls.txt")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "tournament_logs")
OUR_TEAM   = "geoz"   # change if your team name differs


def download_csv(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        print(f"  [ERROR] Could not fetch URL: {e}")
        return None


def detect_our_slot(content: str) -> tuple[int, str, str]:
    """Return (our_slot, our_name, opp_name) by reading the header comment.

    Header format: # Team 0: <name>, Team 1: <name>
    Falls back to slot 0 if unparseable.
    """
    import re
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
    return 0, "us", "opp"  # fallback


def analyze_match(content: str, match_id: str, bot_name: str = ""):
    all_lines = content.splitlines()
    data_lines = [l for l in all_lines if not l.startswith("#")]
    if not data_lines:
        print("  [WARN] Empty file")
        return

    rows = list(csv.DictReader(data_lines))
    if not rows:
        print("  [WARN] No data rows")
        return

    our_slot, our_name, opp_name = detect_our_slot(content)
    our_col = f"team_{our_slot}_bankroll"
    opp_col = f"team_{1 - our_slot}_bankroll"

    last = rows[-1]
    total_hands = int(last.get("hand_number", 0)) + 1
    our_net = float(last.get(our_col, 0))
    opp_net = float(last.get(opp_col, 0))
    ev = our_net / total_hands if total_hands else 0
    result = "WON" if our_net > 0 else "LOST"

    # Per-hand P&L from bankroll changes
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
    breaks = [p for p in pnls if p == 0]

    # Action counts (only our actions: active_team matches our slot)
    our_rows = [r for r in rows if r.get("active_team") == str(our_slot)]
    raises = sum(1 for r in our_rows if r.get("action_type") == "RAISE")
    calls  = sum(1 for r in our_rows if r.get("action_type") == "CALL")
    folds  = sum(1 for r in our_rows if r.get("action_type") == "FOLD")

    bot_tag = f"  [{bot_name}]" if bot_name else ""
    print(f"\n{'='*55}")
    print(f"  Match {match_id}{bot_tag}  —  {result}  ({total_hands} hands)")
    print(f"  {our_name} (us): {our_net:+.0f}  |  {opp_name}: {opp_net:+.0f}")
    print(f"  EV/hand : {ev:+.2f}")
    print(f"  Win rate: {len(wins)}/{len(pnls)} = {len(wins)/len(pnls)*100:.1f}%")
    if wins:
        print(f"  Avg win : +{sum(wins)/len(wins):.1f}  big(>50): {sum(1 for w in wins if w>50)}")
    if losses:
        print(f"  Avg loss: {sum(losses)/len(losses):.1f}  big(<-50): {sum(1 for l in losses if l<-50)}")
    print(f"  Actions : RAISE {raises}  CALL {calls}  FOLD {folds}")
    if "time_used" in last:
        tu = float(last["time_used"])
        print(f"  Time    : {tu:.1f}s  ({tu/total_hands*1000:.0f}ms/hand)")
    print(f"{'='*55}")


def analyze_bot_log(content: str, match_id: str, bot_name: str = ""):
    """Analyze a bot agent log file (.log) from the tournament server."""
    import re
    lines = content.splitlines()

    # Extract agent name from log lines (format: "timestamp - AgentName - LEVEL - msg")
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
        print("  [WARN] No WON/LOST entries found")
        print("  First 5 lines:", lines[:5])
        return

    net = sum(won) - sum(lost)
    wr  = len(won) / total * 100

    # Timing
    ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
    timestamps = [m.group(1) for l in lines for m in [ts_re.match(l)] if m]
    duration = ""
    if len(timestamps) >= 2:
        from datetime import datetime
        t0 = datetime.strptime(timestamps[0],  '%Y-%m-%d %H:%M:%S,%f')
        t1 = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S,%f')
        secs = (t1 - t0).total_seconds()
        duration = f"  Duration: {secs:.0f}s  ({secs/total*1000:.0f}ms/hand)"

    # tl= (time_left) tells us tournament server timing
    tl_vals = [float(m.group(1)) for l in lines for m in [re.search(r'tl=([\d.]+)', l)] if m]
    tl_info = ""
    if tl_vals:
        tl_info = f"  time_left: {min(tl_vals):.0f}s – {max(tl_vals):.0f}s (min seen = {min(tl_vals):.0f}s)"

    # Actions
    raises = sum(1 for l in lines if 'RAISE s=' in l or 'PRE-FLOP RAISE' in l)
    calls  = sum(1 for l in lines if 'CALL  s=' in l)
    folds  = sum(1 for l in lines if 'FOLD  s=' in l or 'PRE-FLOP FOLD' in l)

    bot_label = f"[{agent_name}]" if agent_name else ""
    print(f"\n{'='*55}")
    print(f"  Bot log {match_id}  {bot_label}  ({total} hands)")
    print(f"  Net: {net:+.0f}  EV/h: {net/total:+.2f}")
    print(f"  Win rate: {len(won)}/{total} = {wr:.1f}%")
    if won:   print(f"  Avg win : +{sum(won)/len(won):.1f}")
    if lost:  print(f"  Avg loss: -{sum(lost)/len(lost):.1f}")
    print(f"  Actions : RAISE {raises}  CALL {calls}  FOLD {folds}")
    if duration: print(duration)
    if tl_info:  print(tl_info)
    print(f"{'='*55}")


def parse_watch_file(path: str) -> list[tuple[str, str]]:
    """Parse watch_urls.txt and return list of (url, bot_name) tuples.

    Recognises '# bot: BotName' directives that set the active bot for
    all URLs that follow until the next directive.
    """
    import re
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


def run():
    # Ensure dirs exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(WATCH_FILE):
        open(WATCH_FILE, "w").close()
        print(f"Created {WATCH_FILE}")

    print(f"Watching {WATCH_FILE} for new URLs...")
    print("Tip: add '# bot: PlayerAgent' (or Claude2Agent) before URLs to tag which bot is active.")
    print("Paste presigned S3 URLs into that file (one URL per line) to auto-analyze.\n")

    seen_urls = set()

    # Mark a URL as seen only if its output file already exists on disk.
    # This way, clearing tournament_logs/ causes re-download on next start.
    for url, bot_name in parse_watch_file(WATCH_FILE):
        url_path_init = url.split("?")[0]
        is_log_init = url_path_init.endswith(".log")
        ext_init = ".log" if is_log_init else ".csv"
        mid_init = "?"
        if "match_" in url:
            part = url.split("match_")[-1]
            mid_init = part.split(".")[0].split("?")[0]
        bot_suffix_init = f"_{bot_name}" if bot_name else ""
        fname_init = os.path.join(LOGS_DIR, f"match_{mid_init}{bot_suffix_init}{ext_init}")
        if os.path.exists(fname_init):
            seen_urls.add(url)

    while True:
        try:
            entries = parse_watch_file(WATCH_FILE)
            new_entries = [(url, bot) for url, bot in entries if url not in seen_urls]

            for url, bot_name in new_entries:
                seen_urls.add(url)
                # Extract match ID from URL
                match_id = "?"
                if "match_" in url:
                    part = url.split("match_")[-1]
                    match_id = part.split(".")[0].split("?")[0]

                # Detect file type from URL path
                url_path = url.split("?")[0]
                is_log = url_path.endswith(".log")
                ext = ".log" if is_log else ".csv"

                bot_tag = f" [{bot_name}]" if bot_name else ""
                print(f"\n[NEW] {'Bot log' if is_log else 'Match CSV'} {match_id}{bot_tag} — downloading...")
                content = download_csv(url)
                if content:
                    # Include bot name in filename so files are distinguishable later
                    bot_suffix = f"_{bot_name}" if bot_name else ""
                    fname = os.path.join(LOGS_DIR, f"match_{match_id}{bot_suffix}{ext}")
                    with open(fname, "w") as fout:
                        fout.write(content)
                    print(f"  Saved to {fname}")
                    if is_log:
                        analyze_bot_log(content, match_id, bot_name=bot_name)
                    else:
                        analyze_match(content, match_id, bot_name=bot_name)

        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(2)


if __name__ == "__main__":
    run()
