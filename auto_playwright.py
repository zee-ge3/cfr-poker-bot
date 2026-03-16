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
        browser = p.chromium.launch(headless=False, slow_mo=200)
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
            browser = p.chromium.launch(headless=True)
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
