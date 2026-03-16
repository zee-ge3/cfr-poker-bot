#!/usr/bin/env python3
"""
auto_fetch_logs.py — Automated tournament log fetcher for CMU AI Poker.

Downloads ALL match CSVs and bot logs for team geoz.

The Clerk __session JWT expires in 60s, so this script races the clock:
  Phase 1 (auth, ~15s): Fetch dashboard pages + all presigned S3 URLs
  Phase 2 (no auth):    Download files from S3 using presigned URLs (30min validity)

Usage:
  1. In browser at aipoker.cmudsc.com, open DevTools → Application → Cookies
  2. Copy the __session cookie value
  3. IMMEDIATELY run:
       python3 auto_fetch_logs.py --cookie 'PASTE_HERE'
     (you have 60 seconds from when the cookie was generated)

  Or for repeat use, paste the cookie and pipe it:
       python3 auto_fetch_logs.py --interactive

Other:
  python3 auto_fetch_logs.py --summary     # stats on downloaded logs
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

BASE_URL = "https://aipoker.cmudsc.com"
OUR_TEAM = "geoz"
TEAM_ID = 71
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_logs")
COOKIE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".session_cookie")


# ── Turbo-stream parser ─────────────────────────────────────────────────────

def parse_turbo_stream(raw):
    """Parse Remix v3 turbo-stream format into a Python dict."""
    arr = json.loads(raw)
    if not isinstance(arr, list):
        return None
    # Detect redirect sentinel
    if (len(arr) >= 2 and arr[0] == "SingleFetchRedirect"
            or (isinstance(arr[0], list) and arr[0][0] == "SingleFetchRedirect")):
        return None

    cache = {}

    def resolve(idx, depth=0):
        if depth > 60:
            return None
        if not isinstance(idx, int):
            return idx
        if idx == -5:
            return None
        if idx < 0:
            return idx
        if idx in cache:
            return cache[idx]
        if idx >= len(arr):
            return None
        val = arr[idx]
        if isinstance(val, dict):
            result = {}
            cache[idx] = result
            for k, v in val.items():
                key = resolve(int(k.lstrip("_")), depth + 1)
                value = resolve(v, depth + 1)
                if isinstance(key, str):
                    result[key] = value
            return result
        elif isinstance(val, list):
            result = []
            cache[idx] = result
            for item in val:
                result.append(resolve(item, depth + 1))
            return result
        else:
            cache[idx] = val
            return val

    root = resolve(0)
    if not isinstance(root, dict):
        return None

    # Navigate to the route data
    for route in ("routes/dashboard", "dashboard"):
        if route in root:
            rd = root[route]
            return rd.get("data", rd) if isinstance(rd, dict) else rd
    if "root" in root:
        rd = root["root"]
        return rd.get("data", rd) if isinstance(rd, dict) else rd
    return root


# ── Auth + API ───────────────────────────────────────────────────────────────

def make_headers(cookie):
    return {
        "User-Agent": "Mozilla/5.0 (poker-bot-monitor/2.0)",
        "Cookie": f"__session={cookie}; __client_uat=1",
        "Accept": "*/*",
    }


def fetch_dashboard_page(cookie, page=1, rows=100):
    """Fetch one page of match history. Returns parsed dict or None."""
    qs = f"page={page}&rows={rows}&viewportHeight=900&viewportWidth=1400"
    url = f"{BASE_URL}/dashboard.data?{qs}"
    req = urllib.request.Request(url, headers=make_headers(cookie))
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8")
        if not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
            # Turbo-stream is valid JSON (a list), not a dict
            if isinstance(parsed, list):
                return parse_turbo_stream(raw)
            return parsed
        except json.JSONDecodeError:
            return parse_turbo_stream(raw)
    except urllib.error.HTTPError as e:
        if e.code in (302, 401, 403):
            return None
        raise


def fetch_log_url(cookie, match_id, log_type, team_id=None):
    """Get presigned S3 URL for a match log. Returns URL string or None."""
    url = f"{BASE_URL}/api/logs/{match_id}/{log_type}"
    if log_type == "bot" and team_id:
        url += f"?teamId={team_id}"
    req = urllib.request.Request(url, headers=make_headers(cookie))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("url")
    except Exception:
        return None


def download_s3(url):
    """Download content from a presigned S3 URL (no auth needed)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8")
    except Exception:
        return None


# ── Match analysis ───────────────────────────────────────────────────────────

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


def analyze_csv(content, match_id=""):
    data_lines = [l for l in content.splitlines() if not l.startswith("#")]
    if not data_lines:
        return ""
    rows = list(csv.DictReader(data_lines))
    if not rows:
        return ""

    our_slot, _, opp_name = detect_our_slot(content)
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
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    result = "WON" if our_net > 0 else "LOST"

    return f"{result:4s}  {our_net:+6.0f}  EV {ev:+.2f}  WR {wr:.0f}%  vs {opp_name}"


def existing_match_ids():
    ids = set()
    if not os.path.isdir(LOGS_DIR):
        return ids
    for fname in os.listdir(LOGS_DIR):
        m = re.match(r"match_(\d+).*\.csv$", fname)
        if m:
            ids.add(int(m.group(1)))
    return ids


# ── Main fetch logic ─────────────────────────────────────────────────────────

def run_fetch(cookie, max_pages=None, analyze=True, dry_run=False):
    """Two-phase fetch: grab URLs fast (auth), then download (no auth)."""
    t0 = time.time()
    already = existing_match_ids()

    # ── Phase 1: Fetch match list from dashboard (needs auth) ──────────
    print("Phase 1: Fetching match list...")
    all_matches = []
    page = 1

    while True:
        result = fetch_dashboard_page(cookie, page=page)
        if result is None:
            if page == 1:
                print("  AUTH FAILED — cookie expired or invalid.")
                return 0, 0, False
            break

        matches = result.get("matches", [])
        total_pages = result.get("totalPages", 1)
        total_matches = result.get("totalMatches", 0)
        if page == 1:
            print(f"  {total_matches} matches, {total_pages} pages")
        all_matches.extend(matches)
        if page >= total_pages:
            break
        if max_pages and page >= max_pages:
            break
        page += 1

    # Filter to new matches only
    new_matches = []
    for m in all_matches:
        match_id = m.get("matchId") or (m.get("match", {}) or {}).get("id")
        if match_id and match_id not in already:
            status = ((m.get("match", {}) or {}).get("matchStatus", ""))
            if status in ("COMPLETED", "ERROR", ""):
                new_matches.append(m)

    if not new_matches:
        elapsed = time.time() - t0
        print(f"  All {len(all_matches)} matches already downloaded. ({elapsed:.1f}s)")
        return 0, len(all_matches), True

    print(f"  {len(new_matches)} new matches (have {len(already)})")

    if dry_run:
        for m in new_matches:
            mid = m.get("matchId")
            opp = "?"
            tms = (m.get("match", {}) or {}).get("teamMatches", [])
            for tm in tms:
                if isinstance(tm, dict) and tm.get("teamId") != TEAM_ID:
                    opp = (tm.get("team", {}) or {}).get("name", "?")
            bankroll = m.get("bankroll", 0)
            bot = (m.get("bot", {}) or {}).get("name", "?")
            print(f"    {mid} vs {opp:<20s} {bankroll:+6}  [{bot}]")
        return len(new_matches), len(all_matches), True

    # ── Phase 1b: Grab all presigned URLs concurrently (needs auth) ────
    print(f"  Grabbing presigned URLs ({len(new_matches)*2} requests)...")

    url_tasks = []
    for m in new_matches:
        mid = m.get("matchId")
        tid = m.get("teamId", TEAM_ID)
        if mid:
            url_tasks.append(("csv", mid, tid, m))
            url_tasks.append(("bot", mid, tid, m))

    presigned = {}  # (matchId, type) → s3_url

    def grab_url(task):
        log_type, mid, tid, _ = task
        team_id = tid if log_type == "bot" else None
        url = fetch_log_url(cookie, mid, log_type, team_id)
        return (mid, log_type), url

    workers = min(30, len(url_tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(grab_url, t): t for t in url_tasks}
        done_count = 0
        for future in as_completed(futures):
            key, url = future.result()
            if url:
                presigned[key] = url
            done_count += 1
            if done_count % 50 == 0:
                print(f"    {done_count}/{len(url_tasks)} URLs...")

    elapsed_p1 = time.time() - t0
    csv_urls = sum(1 for k in presigned if k[1] == "csv")
    bot_urls = sum(1 for k in presigned if k[1] == "bot")
    print(f"  Got {csv_urls} CSV + {bot_urls} bot URLs in {elapsed_p1:.1f}s")

    if not presigned:
        print("  No presigned URLs obtained — auth may have expired mid-fetch.")
        return 0, len(all_matches), False

    # ── Phase 2: Download from S3 (no auth needed, 30min validity) ─────
    print(f"\nPhase 2: Downloading files from S3...")
    os.makedirs(LOGS_DIR, exist_ok=True)

    downloaded = 0

    def download_and_save(mid, log_type, s3_url, match_meta):
        content = download_s3(s3_url)
        if not content:
            return None

        if log_type == "csv":
            fpath = os.path.join(LOGS_DIR, f"match_{mid}.csv")
        else:
            bot_name = (match_meta.get("bot", {}) or {}).get("name", "bot")
            fpath = os.path.join(LOGS_DIR, f"match_{mid}_bot_{bot_name}.log")

        with open(fpath, "w") as f:
            f.write(content)
        return (mid, log_type, content, fpath)

    # Build download tasks
    dl_tasks = []
    match_by_id = {}
    for m in new_matches:
        mid = m.get("matchId")
        if mid:
            match_by_id[mid] = m

    for (mid, log_type), s3_url in presigned.items():
        dl_tasks.append((mid, log_type, s3_url, match_by_id.get(mid, {})))

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download_and_save, *t): t for t in dl_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                mid, log_type, content, fpath = result
                if log_type == "csv":
                    downloaded += 1
                    if analyze:
                        analysis = analyze_csv(content, mid)
                        print(f"  [{mid}] {analysis}")
                    elif downloaded % 20 == 0:
                        print(f"  {downloaded} CSVs downloaded...")

    elapsed_total = time.time() - t0
    print(f"\n  Done: {downloaded} new CSVs in {elapsed_total:.1f}s")
    return downloaded, len(all_matches), True


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.isdir(LOGS_DIR):
        print("  No logs downloaded yet.")
        return

    csv_files = sorted(f for f in os.listdir(LOGS_DIR) if f.endswith(".csv"))
    log_files = sorted(f for f in os.listdir(LOGS_DIR) if f.endswith(".log"))
    print(f"\n  Tournament logs: {len(csv_files)} CSVs, {len(log_files)} bot logs")
    print(f"  Directory: {LOGS_DIR}")

    wins, losses = 0, 0
    total_net = 0
    by_opp = defaultdict(lambda: {"w": 0, "l": 0, "net": 0})

    for fname in csv_files:
        fpath = os.path.join(LOGS_DIR, fname)
        try:
            with open(fpath) as f:
                content = f.read()
            our_slot, _, opp = detect_our_slot(content)
            our_col = f"team_{our_slot}_bankroll"
            data_lines = [l for l in content.splitlines() if not l.startswith("#")]
            rows = list(csv.DictReader(data_lines))
            if rows:
                net = float(rows[-1].get(our_col, 0))
                total_net += net
                if net > 0:
                    wins += 1
                    by_opp[opp]["w"] += 1
                else:
                    losses += 1
                    by_opp[opp]["l"] += 1
                by_opp[opp]["net"] += net
        except Exception:
            pass

    total = wins + losses
    if total:
        wr = wins / total * 100
        print(f"  Record: {wins}W-{losses}L ({wr:.1f}% WR)")
        print(f"  Total net: {total_net:+.0f} chips\n")

        print(f"  {'Opponent':<25} {'W':>3}-{'L':<3} {'WR':>6}  {'Net':>8}")
        print(f"  {'─'*50}")
        for opp, d in sorted(by_opp.items(), key=lambda x: x[1]["net"], reverse=True):
            t = d["w"] + d["l"]
            wr = d["w"] / t * 100 if t else 0
            print(f"  {opp:<25} {d['w']:>3}-{d['l']:<3} {wr:>5.1f}%  {d['net']:>+8.0f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_cookie(args_cookie=None):
    raw = None
    if args_cookie:
        raw = args_cookie.strip()
    elif os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE) as f:
            raw = f.read().strip()
    if not raw:
        return None
    if "__session=" in raw:
        for part in raw.split(";"):
            part = part.strip()
            if part.startswith("__session="):
                return part[len("__session="):]
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fetch tournament logs (60s JWT race)")
    parser.add_argument("--cookie", help="__session JWT (paste immediately!)")
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for cookie on stdin")
    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.interactive:
        print("Paste __session cookie value, then press Enter:")
        print("(You have 60s from when the cookie appeared in browser)")
        cookie = input().strip()
    else:
        cookie = load_cookie(args.cookie)

    if not cookie:
        print("No session cookie.")
        print()
        print("Quick usage:")
        print("  1. In browser: aipoker.cmudsc.com → F12 → Application → Cookies")
        print("  2. Copy __session value")
        print("  3. IMMEDIATELY run:")
        print("     python3 auto_fetch_logs.py --cookie 'PASTE'")
        print()
        print("  Or: python3 auto_fetch_logs.py --interactive")
        print("  Or: echo 'JWT' > .session_cookie && python3 auto_fetch_logs.py")
        sys.exit(1)

    # Save for potential re-use (won't work after 60s but harmless)
    with open(COOKIE_FILE, "w") as f:
        f.write(cookie)

    analyze = not args.no_analyze
    os.makedirs(LOGS_DIR, exist_ok=True)

    new, total, auth_ok = run_fetch(
        cookie, max_pages=args.pages,
        analyze=analyze, dry_run=args.dry_run)

    if not auth_ok:
        print("\n  AUTH EXPIRED — JWT only lasts 60s.")
        print("  Copy a fresh __session cookie and run again immediately.")
        sys.exit(1)

    if new and not args.dry_run:
        print(f"\n  Imported {new} new matches")
    print_summary()


if __name__ == "__main__":
    main()
