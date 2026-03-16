# auto_playwright.py
"""
auto_playwright.py — Playwright-based bot submission, session management, and log daemon.

Usage:
  python auto_playwright.py --setup              # one-time interactive login (before bed)
  python auto_playwright.py --submit spy         # swap in spy bot
  python auto_playwright.py --submit main        # swap back main bot
  python auto_playwright.py --daemon             # start overnight fetch loop
  python auto_playwright.py --daemon --interval 15  # custom interval (minutes)
"""
import argparse
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent
BROWSER_STATE = REPO / ".browser_state.json"
OVERNIGHT_LOG = REPO / "overnight_log.jsonl"
AUTH_EXPIRED_FLAG = REPO / ".auth_expired"
SELECTOR_CONFIG = REPO / ".submit_selectors.json"

BASE_URL = "https://aipoker.cmudsc.com"
CLERK_FAPI = "https://clerk.cmudsc.com"   # Clerk Frontend API (derived from pk_live_Y2xlcmsuY211ZHNjLmNvbSQ)
DEFAULT_INTERVAL = 15  # minutes


# ── Zip builder ──────────────────────────────────────────────────────────────

EXCLUDE_PATTERNS = {".DS_Store", "__pycache__", ".pyc", ".pyo", ".pyd",
                    ".git", ".idea", ".vscode"}

def _should_exclude(path: str) -> bool:
    return any(p in path for p in EXCLUDE_PATTERNS)


def build_zip(bot: str, out_path: str) -> None:
    """Build submission zip for 'spy' or 'main' bot.

    When bot='spy':  submission/spy_player.py is zipped AS submission/player.py
    When bot='main': submission/player.py is used as-is
    Both exclude agent_logs/, tournament_logs/, .browser_state.json, etc.
    """
    if bot not in ('spy', 'main'):
        raise ValueError(f"bot must be 'spy' or 'main', got {bot!r}")

    # Determine the source for submission/player.py
    if bot == 'spy':
        player_src = REPO / "submission" / "spy_player.py"
    else:
        player_src = REPO / "submission" / "player.py"

    if not player_src.exists():
        raise FileNotFoundError(f"Source file not found: {player_src}")

    INCLUDE_ROOTS = [
        "gym_env.py", "match.py", "run.py", "agent_test.py",
        "requirements.txt", "agent_config.json",
        "agents/agent.py", "agents/test_agents.py",
    ]

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add fixed root files
        for item in INCLUDE_ROOTS:
            src = REPO / item
            if src.is_file() and not _should_exclude(str(src)):
                zf.write(src, f"aipoker/{item}")
            elif src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file() and not _should_exclude(str(f)):
                        rel = f.relative_to(REPO)
                        zf.write(f, f"aipoker/{rel}")

        # submission/ directory: include all except player.py and spy_player.py
        sub_dir = REPO / "submission"
        for f in sub_dir.iterdir():
            if f.is_file() and f.name != "player.py" and f.name != "spy_player.py":
                if not _should_exclude(str(f)):
                    zf.write(f, f"aipoker/submission/{f.name}")

        # The bot's player.py (renamed if spy)
        zf.write(player_src, "aipoker/submission/player.py")


# ── Playwright session management ────────────────────────────────────────────

def setup():
    """Interactive one-time setup: login + save session state + discover submission selectors."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    print("=" * 60)
    print("  auto_playwright SETUP")
    print("  1. A browser window will open.")
    print("  2. Log into aipoker.cmudsc.com manually.")
    print("  3. Navigate to the bot submission page.")
    print("  4. DO NOT close the browser — press Enter here when ready.")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, slow_mo=200,
            args=['--no-sandbox', '--disable-dev-shm-usage'],
        )
        context = browser.new_context()
        page = context.new_page()
        page.goto(BASE_URL)

        input("\nPress Enter once you have logged in and are on the submission page... ")

        # Save full browser state (cookies + localStorage)
        context.storage_state(path=str(BROWSER_STATE))
        print(f"  Session saved → {BROWSER_STATE}")

        # Discover submission form selectors
        print("\nDiscovering submission page selectors...")
        selectors = _discover_submit_selectors(page)
        if selectors:
            with open(SELECTOR_CONFIG, 'w') as f:
                json.dump(selectors, f, indent=2)
            print(f"  Selectors saved → {SELECTOR_CONFIG}")
            print(f"  Found: {selectors}")
        else:
            print("  WARNING: Could not auto-detect selectors.")
            print("  You will need to set them manually in .submit_selectors.json")
            with open(SELECTOR_CONFIG, 'w') as f:
                json.dump({"file_input": "input[type=file]", "submit_button": "button[type=submit]"}, f)

        browser.close()

    print("\nSetup complete. Run with --submit spy to deploy the spy bot.")


def setup_manual(client_token: str):
    """Headless setup for SSH environments: build session from a manually-provided
    __client cookie value, then discover selectors headlessly.

    How to get the __client token:
      1. Open aipoker.cmudsc.com in your LOCAL browser and log in.
      2. DevTools → Application → Cookies → aipoker.cmudsc.com
      3. Copy the value of the __client cookie (the long one).
      4. Pass it via --setup-manual <value>
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.")
        sys.exit(1)

    client_token = client_token.strip()
    if not client_token:
        print("ERROR: Empty token. Pass the __client cookie value as the argument.")
        sys.exit(1)

    # Build initial browser state with just the __client cookie
    # Playwright expects storage_state format: {cookies: [...], origins: [...]}
    import time as _time
    initial_state = {
        "cookies": [
            {
                "name": "__client",
                "value": client_token,
                "domain": "aipoker.cmudsc.com",
                "path": "/",
                "expires": int(_time.time()) + 86400 * 30,  # 30 days
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            }
        ],
        "origins": [],
    }
    with open(BROWSER_STATE, "w") as f:
        json.dump(initial_state, f, indent=2)
    print(f"  Initial session written → {BROWSER_STATE}")

    # Navigate headlessly to refresh the session and get __session JWT
    print("  Refreshing session headlessly...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage'],
        )
        context = browser.new_context(storage_state=str(BROWSER_STATE))
        page = context.new_page()

        try:
            page.goto(BASE_URL, timeout=15000)
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception as e:
            print(f"  WARNING: Navigation issue ({e}) — proceeding anyway")

        if "sign-in" in page.url or "login" in page.url:
            print("  ERROR: Redirected to login — the __client token may be wrong or expired.")
            print("  Make sure you copied the __client cookie (not __session or another cookie).")
            browser.close()
            BROWSER_STATE.unlink(missing_ok=True)
            sys.exit(1)

        print(f"  Authenticated. Current URL: {page.url}")

        # Refresh session state to capture __session JWT
        context.storage_state(path=str(BROWSER_STATE))
        print(f"  Session saved → {BROWSER_STATE}")

        # Navigate to submit page and discover selectors
        print("  Discovering submission selectors...")
        try:
            page.goto(f"{BASE_URL}/submit", timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)
            selectors = _discover_submit_selectors(page)
        except Exception:
            selectors = {}

        if selectors:
            with open(SELECTOR_CONFIG, "w") as f:
                json.dump(selectors, f, indent=2)
            print(f"  Selectors saved → {SELECTOR_CONFIG}: {selectors}")
        else:
            with open(SELECTOR_CONFIG, "w") as f:
                json.dump({"file_input": "input[type=file]", "submit_button": "button[type=submit]"}, f)
            print("  Could not auto-detect selectors — using defaults.")

        browser.close()

    print("\nSetup complete. Run with --submit spy to deploy the spy bot.")


def _discover_submit_selectors(page) -> dict:
    """Try to find file input and submit button on the current page."""
    selectors = {}
    # Common file input patterns
    for sel in ["input[type='file']", "input[accept]", "input[name*='file']", "input[name*='bot']"]:
        try:
            el = page.query_selector(sel)
            if el:
                selectors['file_input'] = sel
                break
        except Exception:
            pass

    # Common submit button patterns
    for sel in ["button[type='submit']", "button:has-text('Submit')", "button:has-text('Upload')",
                "input[type='submit']", "button:has-text('Deploy')"]:
        try:
            el = page.query_selector(sel)
            if el:
                selectors['submit_button'] = sel
                break
        except Exception:
            pass

    return selectors if len(selectors) == 2 else {}


def submit_bot(bot: str) -> bool:
    """Build zip and upload via Playwright. Returns True on success.

    bot: 'spy' | 'main'
    On failure: prints manual fallback instructions.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.")
        return False

    if not BROWSER_STATE.exists():
        print("ERROR: No saved session. Run --setup first.")
        return False

    if not SELECTOR_CONFIG.exists():
        print("ERROR: No selector config. Run --setup first.")
        return False

    with open(SELECTOR_CONFIG) as f:
        selectors = json.load(f)

    # Build zip to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tf:
        zip_path = tf.name

    try:
        build_zip(bot, zip_path)
        print(f"  Built zip: {zip_path} ({os.path.getsize(zip_path)//1024}KB)")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'],
            )
            context = browser.new_context(storage_state=str(BROWSER_STATE))
            page = context.new_page()

            # Navigate to submission page
            print(f"  Navigating to submission page...")
            page.goto(f"{BASE_URL}/submit", timeout=15000)
            page.wait_for_load_state("networkidle", timeout=15000)

            # Upload file
            file_input_sel = selectors.get('file_input', "input[type='file']")
            try:
                page.set_input_files(file_input_sel, zip_path)
                print(f"  File set on input: {file_input_sel}")
            except Exception as e:
                print(f"  ERROR: Could not set file input ({e})")
                _print_manual_fallback(zip_path, bot)
                browser.close()
                return False

            # Click submit
            submit_sel = selectors.get('submit_button', "button[type='submit']")
            try:
                page.click(submit_sel, timeout=5000)
                page.wait_for_load_state("networkidle", timeout=15000)
                print(f"  Submit clicked. Current URL: {page.url}")
            except Exception as e:
                print(f"  ERROR: Could not click submit ({e})")
                _print_manual_fallback(zip_path, bot)
                browser.close()
                return False

            # Update saved session state (refresh tokens)
            context.storage_state(path=str(BROWSER_STATE))
            browser.close()

        print(f"  Submission complete: bot={bot}")
        return True

    except Exception as e:
        print(f"  ERROR during submission: {e}")
        _print_manual_fallback(zip_path, bot)
        return False
    finally:
        try:
            os.unlink(zip_path)
        except Exception:
            pass


def _print_manual_fallback(zip_path: str, bot: str):
    print("\n  ── MANUAL FALLBACK ──")
    print(f"  Zip file: {zip_path}")
    print(f"  1. Open {BASE_URL} in browser")
    print(f"  2. Navigate to the bot submission page")
    print(f"  3. Upload the zip file manually ({bot} bot)")
    print("  ────────────────────")


# ── Session cookie extraction ─────────────────────────────────────────────────

def _get_client_token() -> str | None:
    """Read the __client token from .browser_state.json."""
    if not BROWSER_STATE.exists():
        return None
    try:
        with open(BROWSER_STATE) as f:
            state = json.load(f)
        for c in state.get("cookies", []):
            if c.get("name") == "__client":
                return c["value"]
    except Exception:
        pass
    return None


def get_session_cookie() -> str | None:
    """Get a fresh __session JWT via Clerk FAPI (no browser required).

    Reads the long-lived __client token from .browser_state.json and calls
    the Clerk Frontend API to obtain a fresh short-lived session JWT.
    Returns the JWT string or None on auth failure.
    """
    import urllib.request

    client_token = _get_client_token()
    if not client_token:
        return None

    try:
        url = f"{CLERK_FAPI}/v1/client?_clerk_js_version=4.73.4"
        req = urllib.request.Request(url, headers={
            "Cookie": f"__client={client_token}",
            "User-Agent": "Mozilla/5.0 (poker-spy-daemon/1.0)",
            "Origin": BASE_URL,
            "Referer": BASE_URL + "/",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        sessions = data.get("response", {}).get("sessions", [])
        if not sessions:
            return None

        # Use the first active session
        active = next((s for s in sessions if s.get("status") == "active"), sessions[0])
        jwt = active.get("last_active_token", {}).get("jwt")
        return jwt or None

    except Exception as e:
        print(f"  [daemon] Clerk API session refresh error: {e}")
        return None


# ── JSONL logger ─────────────────────────────────────────────────────────────

def _log_matches(new_match_ids: set, current_bot: str, leaderboard: list):
    """For each new match ID, parse the CSV and write a JSONL line."""
    import csv as csv_mod

    lb_by_name = {t['name']: t for t in leaderboard}
    logs_dir = REPO / "tournament_logs"

    for mid in new_match_ids:
        csv_path = logs_dir / f"match_{mid}.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path) as f:
                content = f.read()
            lines = content.splitlines()
            header_line = lines[0] if lines else ""
            m = re.match(r"#\s*Team\s*0:\s*(.+?)\s*,\s*Team\s*1:\s*(.+)", header_line)
            if not m:
                continue
            t0, t1 = m.group(1).strip(), m.group(2).strip()
            our_slot = 0 if "geoz" in t0.lower() else 1
            opp_name = t1 if our_slot == 0 else t0
            our_col = f"team_{our_slot}_bankroll"

            data_lines = [l for l in lines if not l.startswith("#")]
            rows = list(csv_mod.DictReader(data_lines))
            if not rows:
                continue
            net = float(rows[-1].get(our_col, 0))
            result = "WON" if net > 0 else "LOST"

            opp_data = lb_by_name.get(opp_name, {})
            record = {
                "ts": datetime.now().isoformat(timespec='seconds'),
                "match_id": mid,
                "bot": current_bot,
                "opponent": opp_name,
                "opp_rank": opp_data.get("rank"),
                "opp_elo": opp_data.get("elo"),
                "result": result,
                "net": net,
            }
            with open(OVERNIGHT_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"  [daemon] Failed to log match {mid}: {e}")


def _fetch_leaderboard() -> list:
    """Fetch leaderboard (no auth needed). Returns list of team dicts."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/leaderboard",
            headers={"User-Agent": "Mozilla/5.0 (poker-spy-daemon/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8")

        teams = []
        for row in re.split(r'<tr class="border-b', html)[1:]:
            rank_m = re.search(r'#<!-- -->(\d+)', row)
            name_m = re.search(r'<span class="transition-colors duration-200">([^<]+)</span>', row)
            nums = re.findall(r'text-right">(\d+)</td>', row)
            wr_m = re.search(r'<span>([\d.]+)<!-- -->%</span>', row)
            if not (rank_m and name_m and wr_m and len(nums) >= 3):
                continue
            teams.append({
                "rank": int(rank_m.group(1)),
                "name": name_m.group(1).strip(),
                "elo": int(nums[1]),
                "matches": int(nums[2]),
            })
        return teams
    except Exception as e:
        print(f"  [daemon] Leaderboard fetch failed: {e}")
        return []


# ── Daemon loop ───────────────────────────────────────────────────────────────

def run_daemon(interval_minutes: int = DEFAULT_INTERVAL, current_bot: str = "spy"):
    """Overnight loop: refresh session, fetch logs, snapshot leaderboard."""
    import auto_fetch_logs

    print(f"Daemon started. Interval: {interval_minutes}m. Bot: {current_bot}")
    print(f"Overnight log: {OVERNIGHT_LOG}")
    print(f"Press Ctrl+C to stop.\n")

    # Clear stale auth flag
    AUTH_EXPIRED_FLAG.unlink(missing_ok=True)

    auth_ok = True
    cycle = 0

    while True:
        cycle += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Cycle {cycle}")

        # Always try leaderboard (no auth required)
        leaderboard = _fetch_leaderboard()
        if leaderboard:
            our = next((t for t in leaderboard if t['name'] == 'geoz'), None)
            if our:
                print(f"  geoz: rank #{our['rank']}  ELO {our['elo']}  matches {our['matches']}")

        # Auth-dependent: log fetch
        cookie = get_session_cookie()
        if not cookie:
            if auth_ok:
                print(f"  [!] Auth expired — log fetching paused. Leaderboard still polling.")
                AUTH_EXPIRED_FLAG.touch()
                print('\a')  # terminal bell
            auth_ok = False
        else:
            if not auth_ok:
                print(f"  [+] Auth restored.")
                AUTH_EXPIRED_FLAG.unlink(missing_ok=True)
            auth_ok = True

            # Get existing match IDs before fetch
            existing_ids = auto_fetch_logs.existing_match_ids()

            # Fetch new logs
            new_count, total, ok = auto_fetch_logs.run_fetch(
                cookie, analyze=False, dry_run=False
            )
            if ok and new_count > 0:
                # Find newly added match IDs
                after_ids = auto_fetch_logs.existing_match_ids()
                new_ids = after_ids - existing_ids
                print(f"  +{new_count} new matches (total {total})")
                _log_matches(new_ids, current_bot, leaderboard)
            elif ok:
                print(f"  No new matches.")

        # Sleep until next cycle
        sleep_secs = interval_minutes * 60
        print(f"  Next cycle in {interval_minutes}m...")
        time.sleep(sleep_secs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Playwright bot automation")
    parser.add_argument("--setup", action="store_true",
                        help="Interactive login + save session + discover selectors")
    parser.add_argument("--setup-manual", metavar="CLIENT_TOKEN",
                        help="Headless setup (SSH-friendly): pass __client cookie value from your browser")
    parser.add_argument("--submit", choices=["spy", "main"],
                        help="Submit spy or main bot")
    parser.add_argument("--daemon", action="store_true",
                        help="Start overnight log fetch daemon")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Daemon interval in minutes (default: 15)")
    parser.add_argument("--bot", choices=["spy", "main"], default="spy",
                        help="Which bot is currently submitted (for daemon JSONL tagging)")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.setup_manual:
        setup_manual(args.setup_manual)
    elif args.submit:
        ok = submit_bot(args.submit)
        sys.exit(0 if ok else 1)
    elif args.daemon:
        run_daemon(interval_minutes=args.interval, current_bot=args.bot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
