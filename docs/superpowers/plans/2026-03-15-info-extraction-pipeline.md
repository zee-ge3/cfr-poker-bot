# Info-Extraction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an overnight intelligence pipeline: a spy bot that collects opponent data, a Playwright daemon that auto-submits bots and fetches logs, and an opponent profiler that outputs ranked exploit notes.

**Architecture:** Three independent components sharing a JSONL file as the data bus. The spy bot plays legally (raise-mode IP, call-mode OOP, with disguise noise), the Playwright daemon handles browser auth + submissions + periodic log fetching, and the profiler reads logs + JSONL to output a ranked opponent table with auto-generated exploit notes.

**Tech Stack:** Python 3.10, Playwright 1.58 (Chromium), existing `gym_env.py` / `agents/agent.py` base, `auto_fetch_logs.run_fetch()`, `treys` evaluator already installed.

**Spec:** `docs/superpowers/specs/2026-03-15-info-extraction-pipeline-design.md`

---

## Chunk 1: Spy Bot

### Files
- Create: `agents/spy_agent.py`
- Create: `submission/spy_player.py`
- Create: `tests/test_spy_agent.py`

---

### Task 1: Spy bot skeleton + position detection

**Files:**
- Create: `agents/spy_agent.py`
- Create: `tests/test_spy_agent.py`

- [ ] **Step 1: Write failing test for position detection**

```python
# tests/test_spy_agent.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.spy_agent import SpyAgent
from gym_env import PokerEnv

AT = PokerEnv.ActionType

def make_obs(blind_position=0, street=0, valid=None, my_bet=1, opp_bet=1,
             min_raise=2, max_raise=200, my_cards=None, opp_discarded=None,
             community=None, opp_last=""):
    if valid is None:
        valid = [False] * 6
        valid[AT.RAISE.value] = True
        valid[AT.CALL.value] = True
        valid[AT.CHECK.value] = True
    return {
        "blind_position": blind_position,
        "street": street,
        "valid_actions": valid,
        "my_bet": my_bet,
        "opp_bet": opp_bet,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "my_cards": my_cards or [0, 1],
        "opp_discarded_cards": opp_discarded or [-1, -1, -1],
        "community_cards": community or [-1, -1, -1, -1, -1],
        "opp_last_action": opp_last,
        "time_used": 0.0,
        "time_left": 500.0,
    }


def test_ip_defaults_to_raise_mode():
    """IP (blind_position=0) bot raises pre-flop at least 70% of the time."""
    import random
    random.seed(42)
    agent = SpyAgent(stream=False)
    raise_count = 0
    n = 200
    for _ in range(n):
        agent._reset_hand_state()
        obs = make_obs(blind_position=0, street=0)
        action = agent.act(obs, 0, False, False, {})
        if action[0] == AT.RAISE.value:
            raise_count += 1
    # Expect ≥70% raise (80% base IP raise mode, 20% flip to call mode)
    assert raise_count / n >= 0.70, f"IP raise rate {raise_count/n:.2%} < 70%"


def test_oop_defaults_to_call_mode():
    """OOP (blind_position=1) bot calls/checks pre-flop at least 70% of the time."""
    import random
    random.seed(99)
    agent = SpyAgent(stream=False)
    call_count = 0
    n = 200
    for _ in range(n):
        agent._reset_hand_state()
        obs = make_obs(blind_position=1, street=0)
        action = agent.act(obs, 0, False, False, {})
        if action[0] in (AT.CALL.value, AT.CHECK.value):
            call_count += 1
    assert call_count / n >= 0.70, f"OOP call/check rate {call_count/n:.2%} < 70%"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_spy_agent.py::test_ip_defaults_to_raise_mode -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'agents.spy_agent'`

- [ ] **Step 3: Create spy_agent.py with position detection and mode selection**

```python
# agents/spy_agent.py
"""SpyAgent — data-extraction bot for CMU AI Poker Tournament.

Strategy: collect opponent intelligence, not chips.
- IP (blind_position=0, SB, acts last post-flop): raise-mode — probe every street
- OOP (blind_position=1, BB, acts first post-flop): call-mode — see showdowns
- 20% mode-flip randomization prevents opponent anti-modeling
"""
import random
from agents.agent import Agent
from gym_env import PokerEnv

AT = PokerEnv.ActionType


class SpyAgent(Agent):

    def __name__(self):
        return "SpyAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._hand_mode = None   # 'raise' | 'call', set on first act() of each hand
        self._prev_street = -1
        self._hand_count = 0

    def _reset_hand_state(self):
        """Reset per-hand state. Called at hand start."""
        self._hand_mode = None
        self._prev_street = -1

    def _select_mode(self, obs) -> str:
        """Choose raise or call mode for this hand based on position."""
        in_position = obs.get("blind_position", 0) == 0   # SB = IP post-flop
        base = 'raise' if in_position else 'call'
        if random.random() < 0.20:
            base = 'call' if base == 'raise' else 'raise'
        return base

    def act(self, obs, reward, terminated, truncated, info):
        # Note: do NOT increment _hand_count here. observe() handles end-of-hand
        # bookkeeping (same pattern as player.py). act() with terminated=True is
        # a terminal signal but observe() is always called after it.
        if terminated:
            return (AT.FOLD.value, 0, 0, 0)

        valid = obs["valid_actions"]
        street = obs["street"]

        # Discard: always keep first two cards (indices 0 and 1)
        if valid[AT.DISCARD.value]:
            return (AT.DISCARD.value, 0, 0, 1)

        # Set mode at the first action of each new hand
        if self._hand_mode is None:
            self._hand_mode = self._select_mode(obs)

        self._prev_street = street
        self.logger.info(
            f"H{self._hand_count} s{street} mode={self._hand_mode} "
            f"IP={obs.get('blind_position',0)==0}"
        )

        if self._hand_mode == 'raise':
            return self._raise_action(obs)
        return self._call_action(obs)

    def _raise_action(self, obs):
        """Raise mode: probe every street with random sizing."""
        valid = obs["valid_actions"]
        # 10% chance to fold instead (disguise — looks like a weak player)
        if valid[AT.FOLD.value] and random.random() < 0.10:
            return (AT.FOLD.value, 0, 0, 0)

        if valid[AT.RAISE.value]:
            pot = obs["my_bet"] + obs["opp_bet"]
            min_r = obs["min_raise"]
            max_r = obs["max_raise"]
            frac = random.uniform(0.3, 1.2)
            amount = max(min_r, min(max_r, int(pot * frac)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def _call_action(self, obs):
        """Call mode: see showdown to collect hand-strength data."""
        valid = obs["valid_actions"]
        # 15% IP-limp disguise: occasionally raise even in call mode
        if valid[AT.RAISE.value] and random.random() < 0.15:
            pot = obs["my_bet"] + obs["opp_bet"]
            min_r = obs["min_raise"]
            max_r = obs["max_raise"]
            amount = max(min_r, min(max_r, int(pot * 0.5)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def observe(self, obs, reward, terminated, truncated, info):
        """Handles end-of-hand bookkeeping (single increment point for _hand_count)."""
        if terminated:
            self._hand_count += 1
            self._reset_hand_state()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_spy_agent.py::test_ip_defaults_to_raise_mode tests/test_spy_agent.py::test_oop_defaults_to_call_mode -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/spy_agent.py tests/test_spy_agent.py
git commit -m "feat: SpyAgent skeleton with IP/OOP mode selection"
```

---

### Task 2: Spy bot discard and edge-case tests

**Files:**
- Modify: `tests/test_spy_agent.py`

- [ ] **Step 1: Add tests for discard handling and mode-flip noise**

Append to `tests/test_spy_agent.py`:

```python
def test_discard_always_handled():
    """When DISCARD is valid, always returns DISCARD action."""
    agent = SpyAgent(stream=False)
    valid = [False] * 6
    valid[AT.DISCARD.value] = True
    obs = make_obs(valid=valid)
    for _ in range(20):
        action = agent.act(obs, 0, False, False, {})
        assert action[0] == AT.DISCARD.value
        assert action[2] == 0  # keep card index 0
        assert action[3] == 1  # keep card index 1


def test_mode_flip_occurs():
    """Roughly 20% of hands flip mode — test over 500 hands."""
    import random
    random.seed(7)
    agent = SpyAgent(stream=False)
    ip_raise_mode = 0
    ip_call_mode = 0
    for _ in range(500):
        agent._reset_hand_state()
        obs = make_obs(blind_position=0, street=0)
        agent._hand_mode = agent._select_mode(obs)
        if agent._hand_mode == 'raise':
            ip_raise_mode += 1
        else:
            ip_call_mode += 1
    # Expect ~80% raise, ~20% call for IP hands
    flip_rate = ip_call_mode / 500
    assert 0.12 <= flip_rate <= 0.28, f"flip rate {flip_rate:.2%} outside [12%, 28%]"


def test_raise_sizing_varies():
    """Raise amounts vary across calls (not always same size)."""
    import random
    random.seed(1)
    agent = SpyAgent(stream=False)
    agent._hand_mode = 'raise'
    obs = make_obs(blind_position=0, street=2, my_bet=10, opp_bet=10,
                   min_raise=2, max_raise=200)
    amounts = set()
    for _ in range(30):
        action = agent.act(obs, 0, False, False, {})
        if action[0] == AT.RAISE.value:
            amounts.add(action[1])
    assert len(amounts) >= 3, f"Only {len(amounts)} distinct raise sizes — not varying enough"
```

- [ ] **Step 2: Run new tests**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_spy_agent.py -v
```
Expected: `5 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_spy_agent.py
git commit -m "test: spy agent discard, mode-flip noise, raise sizing tests"
```

---

### Task 3: Create standalone submission/spy_player.py

The tournament server discovers agents from `submission/player.py`. When submitting the spy bot, `spy_player.py` is zipped *as* `player.py` (the `auto_playwright.py --submit spy` command handles this rename). The file itself is a self-contained copy of the SpyAgent logic — no imports from `agents/spy_agent.py` — matching the pattern of `submission/player.py`.

**Files:**
- Create: `submission/spy_player.py`

- [ ] **Step 1: Create spy_player.py as a standalone file**

Copy `agents/spy_agent.py` content into `submission/spy_player.py`. The imports are identical to `submission/player.py` — both use `from agents.agent import Agent` and `from gym_env import PokerEnv` (both files are included in the tournament zip):

```python
# submission/spy_player.py
"""
SpyAgent — data-extraction bot. Submitted temporarily to collect opponent intelligence.
Sole goal: maximize information yield, not chips.
"""
import random

from agents.agent import Agent
from gym_env import PokerEnv

AT = PokerEnv.ActionType


class PlayerAgent(Agent):
    """Named PlayerAgent so tournament server picks it up correctly."""

    def __name__(self):
        return "SpyAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._hand_mode = None
        self._prev_street = -1
        self._hand_count = 0

    def _reset_hand_state(self):
        self._hand_mode = None
        self._prev_street = -1

    def _select_mode(self, obs) -> str:
        in_position = obs.get("blind_position", 0) == 0
        base = 'raise' if in_position else 'call'
        if random.random() < 0.20:
            base = 'call' if base == 'raise' else 'raise'
        return base

    def act(self, obs, reward, terminated, truncated, info):
        # observe() handles _hand_count increment — do NOT increment here
        if terminated:
            return (AT.FOLD.value, 0, 0, 0)

        valid = obs["valid_actions"]
        street = obs["street"]

        if valid[AT.DISCARD.value]:
            return (AT.DISCARD.value, 0, 0, 1)

        if self._hand_mode is None:
            self._hand_mode = self._select_mode(obs)

        self._prev_street = street
        self.logger.info(
            f"H{self._hand_count} s{street} mode={self._hand_mode} "
            f"IP={obs.get('blind_position',0)==0}"
        )

        if self._hand_mode == 'raise':
            return self._raise_action(obs)
        return self._call_action(obs)

    def _raise_action(self, obs):
        valid = obs["valid_actions"]
        if valid[AT.FOLD.value] and random.random() < 0.10:
            return (AT.FOLD.value, 0, 0, 0)
        if valid[AT.RAISE.value]:
            pot = obs["my_bet"] + obs["opp_bet"]
            frac = random.uniform(0.3, 1.2)
            amount = max(obs["min_raise"], min(obs["max_raise"], int(pot * frac)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def _call_action(self, obs):
        valid = obs["valid_actions"]
        if valid[AT.RAISE.value] and random.random() < 0.15:
            pot = obs["my_bet"] + obs["opp_bet"]
            amount = max(obs["min_raise"], min(obs["max_raise"], int(pot * 0.5)))
            return (AT.RAISE.value, amount, 0, 0)
        if valid[AT.CALL.value]:
            return (AT.CALL.value, 0, 0, 0)
        if valid[AT.CHECK.value]:
            return (AT.CHECK.value, 0, 0, 0)
        return (AT.FOLD.value, 0, 0, 0)

    def observe(self, obs, reward, terminated, truncated, info):
        """Single increment point for _hand_count."""
        if terminated:
            self._hand_count += 1
            self._reset_hand_state()
```

- [ ] **Step 2: Verify it runs against test agents**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "
from submission.spy_player import PlayerAgent
from agents.test_agents import FoldAgent
from gym_env import PokerEnv
env = PokerEnv()
spy = PlayerAgent(stream=False)
fold = FoldAgent(stream=False)
obs, info = env.reset()
print('SpyAgent smoke test: OK')
"
```
Expected: `SpyAgent smoke test: OK`

- [ ] **Step 3: Commit**

```bash
git add submission/spy_player.py
git commit -m "feat: spy_player.py submission version (standalone, named PlayerAgent)"
```

---

## Chunk 2: Playwright Automation Daemon

### Files
- Create: `auto_playwright.py`
- Modify: `.gitignore` (add browser state files)

---

### Task 4: Zip builder (no AWS dependency)

The `create_release.sh` script calls `aws s3 cp` which fails without credentials. `auto_playwright.py` reimplements the zip logic inline.

**Files:**
- Create: `auto_playwright.py` (zip section only for now)
- Create: `tests/test_auto_playwright.py`

- [ ] **Step 1: Write failing test for zip builder**

```python
# tests/test_auto_playwright.py
import os, sys, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_build_spy_zip_contains_correct_player():
    """spy zip: submission/player.py content is spy_player.py content."""
    from auto_playwright import build_zip
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    try:
        build_zip('spy', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            assert any('submission/player.py' in n for n in names), \
                f"submission/player.py not in zip: {names}"
            content = z.read([n for n in names if n.endswith('submission/player.py')][0]).decode()
            assert 'SpyAgent' in content, "spy zip player.py doesn't contain SpyAgent"
    finally:
        os.unlink(zip_path)


def test_build_main_zip_contains_main_player():
    """main zip: submission/player.py content is the real player.py."""
    from auto_playwright import build_zip
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    try:
        build_zip('main', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            content = z.read([n for n in names if n.endswith('submission/player.py')][0]).decode()
            assert 'SpyAgent' not in content, "main zip shouldn't contain SpyAgent"
    finally:
        os.unlink(zip_path)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_auto_playwright.py::test_build_spy_zip_contains_correct_player -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'auto_playwright'`

- [ ] **Step 3: Create auto_playwright.py with build_zip only**

```python
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

        # submission/ directory: include all except player.py (we control that)
        sub_dir = REPO / "submission"
        for f in sub_dir.iterdir():
            if f.is_file() and f.name != "player.py" and f.name != "spy_player.py":
                if not _should_exclude(str(f)):
                    zf.write(f, f"aipoker/submission/{f.name}")

        # The bot's player.py (renamed if spy)
        zf.write(player_src, "aipoker/submission/player.py")
```

- [ ] **Step 4: Run tests**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_auto_playwright.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add auto_playwright.py tests/test_auto_playwright.py
git commit -m "feat: auto_playwright build_zip with spy/main switching"
```

---

### Task 5: Playwright setup — interactive login + selector discovery

**Files:**
- Modify: `auto_playwright.py` (add `setup()` function)

- [ ] **Step 1: Add setup() function to auto_playwright.py**

Append to `auto_playwright.py`:

```python
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
```

- [ ] **Step 2: Verify setup() is importable and CLI-parseable**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "from auto_playwright import setup, build_zip; print('imports OK')"
```
Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add auto_playwright.py
git commit -m "feat: auto_playwright setup() — interactive login + selector discovery"
```

---

### Task 6: Bot submission via Playwright

**Files:**
- Modify: `auto_playwright.py` (add `submit_bot()`)

- [ ] **Step 1: Add submit_bot() function to auto_playwright.py**

Append to `auto_playwright.py`:

```python
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
```

- [ ] **Step 2: Verify no import errors**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "from auto_playwright import submit_bot; print('submit_bot importable')"
```
Expected: `submit_bot importable`

- [ ] **Step 3: Commit**

```bash
git add auto_playwright.py
git commit -m "feat: auto_playwright submit_bot() — headless zip upload via Playwright"
```

---

### Task 7: Daemon — session refresh + log fetch loop

**Files:**
- Modify: `auto_playwright.py` (add `get_session_cookie()`, `run_daemon()`)
- Modify: `.gitignore`

- [ ] **Step 1: Update .gitignore**

```bash
cd /home/g30rgez/poker/poker-engine-2026
cat >> .gitignore << 'EOF'
.browser_state.json
.session_cookie
.auth_expired
.submit_selectors.json
overnight_log.jsonl
EOF
```

- [ ] **Step 2: Add daemon functions to auto_playwright.py**

Append to `auto_playwright.py`:

```python
# ── Session cookie extraction ─────────────────────────────────────────────────

def get_session_cookie() -> str | None:
    """Load saved browser state, navigate to dashboard to refresh Clerk JWT,
    extract __session cookie. Returns cookie value or None on auth failure."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    if not BROWSER_STATE.exists():
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(BROWSER_STATE))
            page = context.new_page()

            # Navigate to dashboard — Clerk client auto-refreshes __session JWT
            page.goto(f"{BASE_URL}/dashboard.data?page=1&rows=1", timeout=15000)
            page.wait_for_load_state("networkidle", timeout=10000)

            # Check for redirect (auth expired)
            if "sign-in" in page.url or "login" in page.url:
                browser.close()
                return None

            # Extract __session cookie
            cookies = context.cookies()
            session_cookie = next(
                (c["value"] for c in cookies if c["name"] == "__session"),
                None
            )

            # Persist refreshed state
            context.storage_state(path=str(BROWSER_STATE))
            browser.close()
            return session_cookie

    except Exception as e:
        print(f"  [daemon] Session refresh error: {e}")
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
            m = re.match(r"#\s*Team\s*0:\s*(.+?),\s*Team\s*1:\s*(.+)", header_line)
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
```

- [ ] **Step 3: Add CLI entry point to auto_playwright.py**

Append to `auto_playwright.py`:

```python
# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Playwright bot automation")
    parser.add_argument("--setup", action="store_true",
                        help="Interactive login + save session + discover selectors")
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
    elif args.submit:
        ok = submit_bot(args.submit)
        sys.exit(0 if ok else 1)
    elif args.daemon:
        run_daemon(interval_minutes=args.interval, current_bot=args.bot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify daemon imports and CLI help works**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python auto_playwright.py --help
```
Expected: output shows `--setup`, `--submit`, `--daemon`, `--interval`, `--bot` flags with no errors.

- [ ] **Step 5: Test daemon leaderboard fetch independently**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "
from auto_playwright import _fetch_leaderboard
teams = _fetch_leaderboard()
if teams:
    print(f'Leaderboard OK: {len(teams)} teams')
    geoz = next((t for t in teams if t['name'] == 'geoz'), None)
    print(f'geoz: {geoz}')
else:
    print('No teams returned — check network')
"
```
Expected: prints team count and geoz's current stats.

- [ ] **Step 6: Run zip tests still pass**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_auto_playwright.py -v
```
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add auto_playwright.py .gitignore
git commit -m "feat: auto_playwright daemon — session refresh, log fetch loop, leaderboard snapshot"
```

---

## Chunk 3: Opponent Profiler

### Files
- Create: `opponent_profiler.py`
- Create: `tests/test_opponent_profiler.py`

---

### Task 8: CSV parsing — position derivation and action sequences

**Files:**
- Create: `opponent_profiler.py` (parse section)
- Create: `tests/test_opponent_profiler.py`

- [ ] **Step 1: Write failing tests for CSV parsing**

```python
# tests/test_opponent_profiler.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from io import StringIO


SAMPLE_CSV = """# Team 0: opponent_bot, Team 1: geoz
hand_number,street,active_team,team_0_bankroll,team_1_bankroll,action_type,action_amount,action_keep_1,action_keep_2,team_0_cards,team_1_cards,board_cards,team_0_discarded,team_1_discarded,team_0_bet,team_1_bet
0,Pre-Flop,0,0,0,RAISE,4,0,0,[],[],[],[],[],1,2
0,Pre-Flop,1,0,0,CALL,0,0,0,[],[],[],[],[],2,2
0,Flop,1,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2
0,Flop,0,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2
0,Flop,1,0,0,CHECK,0,0,0,[],[],[],[],[],2,2
0,Flop,0,0,0,RAISE,6,0,0,[],[],[],[],[],2,2
0,Flop,1,0,0,FOLD,0,0,0,[],[],[],[],[],2,2
1,Pre-Flop,1,6,-6,RAISE,4,0,0,[],[],[],[],[],1,2
1,Pre-Flop,0,6,-6,CALL,0,0,0,[],[],[],[],[],2,2
1,Flop,0,6,-6,DISCARD,0,0,1,[],[],[],[],[],2,2
1,Flop,1,6,-6,DISCARD,0,0,1,[],[],[],[],[],2,2
1,Flop,0,6,-6,RAISE,4,0,0,[],[],[],[],[],2,2
1,Flop,1,6,-6,CALL,0,0,0,[],[],[],[],[],2,2
"""


def _parse_csv(content):
    from opponent_profiler import parse_match_csv
    return parse_match_csv(content)


def test_geoz_slot_detection():
    """Correctly identifies geoz as team 1."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['geoz_slot'] == 1
    assert result['opp_name'] == 'opponent_bot'


def test_position_derivation_hand0():
    """Hand 0: first Pre-Flop actor is team 0 → team 0 is SB (IP post-flop)."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['hands'][0]['opp_is_ip'] == True  # opp=team0=SB=IP


def test_position_derivation_hand1():
    """Hand 1: first Pre-Flop actor is team 1 (geoz) → opp (team 0) is BB = OOP."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['hands'][1]['opp_is_ip'] == False  # opp=team0=BB=OOP


def test_fold_to_raise_detected():
    """Hand 1 Flop: geoz (team 1) raised, opp (team 0) called → NOT a fold event.
    Append a hand where geoz raises and opp folds to confirm fold detection works."""
    # Hand 2: geoz raises flop, opp folds
    csv_with_fold = SAMPLE_CSV + (
        "2,Pre-Flop,0,0,0,RAISE,4,0,0,[],[],[],[],[],1,2\n"
        "2,Pre-Flop,1,0,0,CALL,0,0,0,[],[],[],[],[],2,2\n"
        "2,Flop,1,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "2,Flop,0,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "2,Flop,1,0,0,RAISE,6,0,0,[],[],[],[],[],2,2\n"
        "2,Flop,0,0,0,FOLD,0,0,0,[],[],[],[],[],2,2\n"
    )
    result = _parse_csv(csv_with_fold)
    ftr_events = result['opp_ftr_events']
    # Hand 1 Flop: geoz raised, opp called → folded=False
    # Hand 2 Flop: geoz raised, opp folded → folded=True
    assert any(e['folded'] for e in ftr_events), \
        "Expected at least one fold-to-raise event"
    assert any(not e['folded'] for e in ftr_events), \
        "Expected at least one non-fold response to raise"
    # Both should be on Flop street
    assert all(e['street'] == 'Flop' for e in ftr_events)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_opponent_profiler.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'opponent_profiler'`

- [ ] **Step 3: Create opponent_profiler.py with parse_match_csv()**

```python
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
        pf_rows = [r for r in h_rows if r['street'] == 'Pre-Flop']
        opp_is_ip = None
        if pf_rows:
            first_actor = int(pf_rows[0]['active_team'])
            opp_is_ip = (first_actor == opp_slot)  # opp is SB → IP

        hand_info = {
            'hand_num': hand_num,
            'opp_is_ip': opp_is_ip,
        }
        hands.append(hand_info)

        # Find fold-to-raise events: we raised, did opp fold?
        for street in STREETS:
            s_rows = [r for r in h_rows if r['street'] == street
                      and r['action_type'] not in ('DISCARD',)]
            # Find sequences: geoz raises, then opp acts
            for i, row in enumerate(s_rows):
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
                            break

    return {
        'geoz_slot': geoz_slot,
        'opp_name': opp_name,
        'hands': hands,
        'opp_ftr_events': opp_ftr_events,
        'rows': rows,
        'opp_slot': opp_slot,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_opponent_profiler.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add opponent_profiler.py tests/test_opponent_profiler.py
git commit -m "feat: opponent_profiler parse_match_csv with position derivation + FTR events"
```

---

### Task 9: Per-opponent metrics aggregation

**Files:**
- Modify: `opponent_profiler.py` (add aggregation)
- Modify: `tests/test_opponent_profiler.py` (add metrics tests)

- [ ] **Step 1: Add metrics tests**

Append to `tests/test_opponent_profiler.py`:

```python
def test_ftr_aggregation():
    """FTR rate computed correctly from multiple hands."""
    from opponent_profiler import aggregate_opponent

    ftr_events = [
        {'street': 'Flop', 'opp_is_ip': False, 'folded': True},
        {'street': 'Flop', 'opp_is_ip': False, 'folded': True},
        {'street': 'Flop', 'opp_is_ip': False, 'folded': False},
        {'street': 'Turn', 'opp_is_ip': True,  'folded': False},
    ]
    match_data = {
        'opp_name': 'TestBot',
        'hands': [{'hand_num': i, 'opp_is_ip': False} for i in range(120)],
        'opp_ftr_events': ftr_events,
        'rows': [],
        'opp_slot': 0,
        'geoz_slot': 1,
    }
    profile = aggregate_opponent([match_data])
    # Flop OOP: 2 folds out of 3 raises = 66.7%
    assert abs(profile['ftr_oop_Flop'] - 2/3) < 0.01
    # Turn IP: 0 folds out of 1 = 0%
    assert profile['ftr_ip_Turn'] == 0.0
    assert profile['total_hands'] == 120


def test_pf_fold_rate():
    """Pre-flop fold rate computed from action rows."""
    from opponent_profiler import aggregate_opponent, parse_match_csv
    result = _parse_csv(SAMPLE_CSV)
    profile = aggregate_opponent([result])
    # Hand 0: opp (team 0) did not fold pre-flop (raised)
    # Hand 1: opp (team 0) called pre-flop
    assert profile['pf_fold_rate'] == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_opponent_profiler.py::test_ftr_aggregation -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'aggregate_opponent'`

- [ ] **Step 3: Add aggregate_opponent() to opponent_profiler.py**

Append to `opponent_profiler.py`:

```python
# ── Metrics aggregation ────────────────────────────────────────────────────────

def aggregate_opponent(match_data_list: list) -> dict:
    """Aggregate multiple parsed match dicts into a single opponent profile."""
    total_hands = 0
    ftr_counts = defaultdict(lambda: {'fold': 0, 'total': 0})
    pf_fold_total = 0
    pf_action_total = 0

    for md in match_data_list:
        if not md:
            continue
        total_hands += len(md.get('hands', []))

        # FTR events
        for ev in md.get('opp_ftr_events', []):
            pos = 'ip' if ev['opp_is_ip'] else 'oop'
            key = f"{pos}_{ev['street']}"
            ftr_counts[key]['total'] += 1
            if ev['folded']:
                ftr_counts[key]['fold'] += 1

        # Pre-flop fold rate from raw rows
        opp_slot = md.get('opp_slot', 0)
        for row in md.get('rows', []):
            if row['street'] == 'Pre-Flop' and int(row['active_team']) == opp_slot:
                pf_action_total += 1
                if row['action_type'] == 'FOLD':
                    pf_fold_total += 1

    # Build profile
    profile = {'total_hands': total_hands}
    profile['pf_fold_rate'] = pf_fold_total / pf_action_total if pf_action_total else 0.0

    for pos in ('oop', 'ip'):
        for street in STREETS:
            key = f"{pos}_{street}"
            d = ftr_counts[key]
            val = d['fold'] / d['total'] if d['total'] >= 3 else None
            profile[f"ftr_{key}"] = val if val is not None else float('nan')
            profile[f"ftr_{key}_n"] = d['total']

    return profile
```

- [ ] **Step 4: Run tests**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_opponent_profiler.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add opponent_profiler.py tests/test_opponent_profiler.py
git commit -m "feat: aggregate_opponent() — FTR rates, PF fold rate per opponent"
```

---

### Task 10: Opponent type classification + exploit note generation

**Files:**
- Modify: `opponent_profiler.py` (add classification + exploit notes)

- [ ] **Step 1: Add classification test**

Append to `tests/test_opponent_profiler.py`:

```python
def test_opponent_type_classification():
    from opponent_profiler import classify_opponent_type
    # Maniac: very low PF fold, very high raise rate → classify as 'maniac'
    assert classify_opponent_type(pf_fold_rate=0.05, vpip=0.95) == 'maniac'
    # LAG: moderate-low PF fold, high VPIP
    assert classify_opponent_type(pf_fold_rate=0.25, vpip=0.75) == 'lag'
    # TAG: moderate PF fold
    assert classify_opponent_type(pf_fold_rate=0.45, vpip=0.55) == 'tag'
    # Calling station: high VPIP, low raise rate
    assert classify_opponent_type(pf_fold_rate=0.05, vpip=0.95, raise_rate=0.05) == 'calling_station'
```

- [ ] **Step 2: Add classify_opponent_type() and exploit note generator**

Append to `opponent_profiler.py`:

```python
# ── Opponent type classification ───────────────────────────────────────────────

def classify_opponent_type(pf_fold_rate: float, vpip: float,
                           raise_rate: float = 0.5) -> str:
    """Classify opponent into TAG/LAG/maniac/calling_station."""
    if vpip >= 0.80 and raise_rate < 0.15:
        return 'calling_station'
    if pf_fold_rate <= 0.15:
        return 'maniac'
    if pf_fold_rate <= 0.35:
        return 'lag'
    return 'tag'


def generate_exploit_note(opp_name: str, rank: int, profile: dict) -> str:
    """Generate a one-line exploit note from an opponent profile."""
    notes = []

    # High OOP fold-to-raise (post-flop)
    for street in ('Flop', 'Turn', 'River'):
        rate = profile.get(f'ftr_oop_{street}', float('nan'))
        n = profile.get(f'ftr_oop_{street}_n', 0)
        if n >= 10 and rate >= 0.65:
            notes.append(f"folds {rate:.0%} OOP on {street} → raise every {street} when they're OOP")
            break

    # High IP fold-to-raise
    for street in ('Flop', 'Turn'):
        rate = profile.get(f'ftr_ip_{street}', float('nan'))
        n = profile.get(f'ftr_ip_{street}_n', 0)
        if n >= 10 and rate >= 0.55:
            notes.append(f"also folds {rate:.0%} IP on {street} → probe aggressively")
            break

    # Very low PF fold → blind steals useless
    pf = profile.get('pf_fold_rate', 0.5)
    if pf < 0.10:
        notes.append("never folds PF → skip blind steals, raise post-flop instead")

    if not notes:
        notes.append("no strong exploit signal yet")

    rank_str = f"(#{rank})" if rank else ""
    return f"{opp_name} {rank_str}: " + "; ".join(notes)
```

- [ ] **Step 3: Run all tests**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/test_opponent_profiler.py -v
```
Expected: `7 passed`

- [ ] **Step 4: Commit**

```bash
git add opponent_profiler.py tests/test_opponent_profiler.py
git commit -m "feat: opponent type classification + exploit note generation"
```

---

### Task 11: Main profiler report — JSONL loading + ranked table output

**Files:**
- Modify: `opponent_profiler.py` (add `run_report()` and CLI)

- [ ] **Step 1: Add run_report() and main()**

Append to `opponent_profiler.py`:

```python
# ── Report runner ─────────────────────────────────────────────────────────────

def load_spy_match_ids() -> set:
    """Load match IDs that were played by the spy bot from overnight_log.jsonl."""
    ids = set()
    if not OVERNIGHT_LOG.exists():
        return ids
    with open(OVERNIGHT_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get('bot') == 'spy':
                    ids.add(rec['match_id'])
            except json.JSONDecodeError:
                pass
    return ids


def load_leaderboard_ranks() -> dict:
    """Load most recent rank per opponent from overnight_log.jsonl."""
    ranks = {}  # opp_name → {rank, elo}
    if not OVERNIGHT_LOG.exists():
        return ranks
    with open(OVERNIGHT_LOG) as f:
        for line in f:
            try:
                rec = json.loads(line)
                opp = rec.get('opponent')
                if opp and rec.get('opp_rank') is not None:
                    ranks[opp] = {'rank': rec['opp_rank'], 'elo': rec.get('opp_elo')}
            except json.JSONDecodeError:
                pass
    return ranks


def run_report(min_hands: int = MIN_HANDS, top_n: int = None):
    """Load spy-bot match logs, compute profiles, print ranked table + exploit notes."""
    spy_ids = load_spy_match_ids()
    ranks = load_leaderboard_ranks()

    if not spy_ids:
        print("No spy-bot match IDs found in overnight_log.jsonl.")
        print("Either the daemon hasn't run yet or --bot spy wasn't specified.")
        return

    print(f"Loading {len(spy_ids)} spy-bot match CSVs from {LOGS_DIR}...")

    # Parse all spy match CSVs, group by opponent
    by_opp = defaultdict(list)
    missing = 0
    for mid in spy_ids:
        csv_path = LOGS_DIR / f"match_{mid}.csv"
        if not csv_path.exists():
            missing += 1
            continue
        with open(csv_path) as f:
            content = f.read()
        md = parse_match_csv(content)
        if md:
            by_opp[md['opp_name']].append(md)

    if missing:
        print(f"  Warning: {missing} CSVs not found in {LOGS_DIR}")

    # Aggregate profiles
    profiles = {}
    for opp_name, matches in by_opp.items():
        p = aggregate_opponent(matches)
        p['match_count'] = len(matches)
        profiles[opp_name] = p

    # Filter by min hands
    qualified = {k: v for k, v in profiles.items() if v['total_hands'] >= min_hands}
    print(f"  {len(profiles)} opponents found, {len(qualified)} with ≥{min_hands} hands\n")

    # Sort by leaderboard rank (unranked last)
    def sort_key(item):
        opp, _ = item
        r = ranks.get(opp, {}).get('rank')
        return r if r is not None else 9999

    sorted_opps = sorted(qualified.items(), key=sort_key)
    if top_n:
        sorted_opps = sorted_opps[:top_n]

    # Print table
    import math
    header = f"{'Rank':<5} {'Opponent':<22} {'PF-fold':>7} {'FTR-OOP-Flop':>12} {'FTR-IP-Flop':>11} {'Type':<14} {'Hands':>6}"
    print(header)
    print("─" * len(header))

    for opp_name, profile in sorted_opps:
        rank_info = ranks.get(opp_name, {})
        rank = rank_info.get('rank', '?')
        pf = profile.get('pf_fold_rate', float('nan'))
        ftr_oop = profile.get('ftr_oop_Flop', float('nan'))
        ftr_ip = profile.get('ftr_ip_Flop', float('nan'))

        # Compute VPIP proxy from PF fold rate
        vpip = 1.0 - pf
        opp_type = classify_opponent_type(pf_fold_rate=pf, vpip=vpip)

        pf_str = f"{pf:.0%}" if not math.isnan(pf) else "  ?"
        oop_str = f"{ftr_oop:.0%}" if not math.isnan(ftr_oop) else "  ?"
        ip_str = f"{ftr_ip:.0%}" if not math.isnan(ftr_ip) else "  ?"

        print(f"#{rank!s:<4} {opp_name:<22} {pf_str:>7} {oop_str:>12} {ip_str:>11} {opp_type:<14} {profile['total_hands']:>6}")

    # Exploit notes
    print(f"\n{'─'*60}")
    print("EXPLOIT NOTES")
    print("─" * 60)
    for opp_name, profile in sorted_opps:
        rank = ranks.get(opp_name, {}).get('rank')
        note = generate_exploit_note(opp_name, rank, profile)
        print(note)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Opponent profiler — overnight log analysis")
    parser.add_argument("--min-hands", type=int, default=MIN_HANDS,
                        help=f"Minimum hands to include opponent (default: {MIN_HANDS})")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top-N ranked opponents")
    args = parser.parse_args()
    run_report(min_hands=args.min_hands, top_n=args.top)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with existing logs (no spy JSONL needed)**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python opponent_profiler.py --min-hands 1
```
Expected: prints "No spy-bot match IDs found in overnight_log.jsonl" — correct behavior since no overnight run has happened yet.

- [ ] **Step 3: Integration test — manually inject test JSONL entry and verify report**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "
import json, os
from pathlib import Path

# Pick a real existing match CSV and pretend it was a spy match
logs_dir = Path('tournament_logs')
csvs = list(logs_dir.glob('match_*.csv'))
if csvs:
    # Get a match ID from filename
    import re
    mid = int(re.search(r'match_(\d+)\.csv', csvs[0].name).group(1))
    with open('overnight_log.jsonl', 'a') as f:
        f.write(json.dumps({'ts': '2026-03-15T23:00:00', 'match_id': mid, 'bot': 'spy',
                            'opponent': 'TestOpp', 'opp_rank': 5, 'opp_elo': 1700,
                            'result': 'WON', 'net': 100}) + '\n')
    print(f'Injected match_id={mid} as spy match')
else:
    print('No CSV files found in tournament_logs/')
"
python opponent_profiler.py --min-hands 1
```
Expected: Shows a table with at least the injected opponent. May show "?" for some metrics depending on match content — that's fine.

- [ ] **Step 4: Clean up test JSONL entry**

```bash
cd /home/g30rgez/poker/poker-engine-2026
# Remove the test entry (last line of overnight_log.jsonl)
python -c "
lines = open('overnight_log.jsonl').readlines() if __import__('os').path.exists('overnight_log.jsonl') else []
test_lines = [l for l in lines if 'TestOpp' not in l]
if test_lines != lines:
    open('overnight_log.jsonl', 'w').writelines(test_lines)
    print('Cleaned test entry')
else:
    print('Nothing to clean')
"
```

- [ ] **Step 5: Run full test suite**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/ -v
```
Expected: All tests pass (`test_spy_agent.py` + `test_auto_playwright.py` + `test_opponent_profiler.py`).

- [ ] **Step 6: Commit**

```bash
git add opponent_profiler.py
git commit -m "feat: opponent_profiler run_report() — ranked table + exploit notes from JSONL"
```

---

## Chunk 4: Integration + Pre-Flight Checklist

### Task 12: End-to-end pre-flight test

This task validates the full overnight workflow before first deployment.

**Files:** No new files.

- [ ] **Step 1: Verify spy bot plays a complete local match**

```bash
cd /home/g30rgez/poker/poker-engine-2026
# Edit agent_config.json to use spy agent temporarily
python -c "
import json
cfg = json.load(open('agent_config.json'))
print('Current config:', json.dumps(cfg, indent=2))
"
```
Note the current config. Then update temporarily:

```bash
python -c "
import json
cfg = json.load(open('agent_config.json'))
# Save backup
json.dump(cfg, open('agent_config.json.bak', 'w'), indent=2)
# Point bot 0 to spy agent
cfg_new = {k: v for k, v in cfg.items()}
# Set first agent to spy — check actual key names from output above
print('Keys:', list(cfg.keys()))
"
```
Open `agent_config.json`, set one agent path to `agents/spy_agent.py`, run match, restore config:

```bash
python run.py  # runs 1000-hand match
# Verify it completes without errors, then restore
cp agent_config.json.bak agent_config.json
```
Expected: Match runs to completion, no Python errors, bot responds within time limits.

- [ ] **Step 2: Verify zip builds correctly**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "
from auto_playwright import build_zip
import tempfile, zipfile, os
with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
    zip_path = f.name
build_zip('spy', zip_path)
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    print('Zip contents:')
    for n in sorted(names):
        print(' ', n)
    player_content = z.read([n for n in names if n.endswith('submission/player.py')][0]).decode()[:100]
    print('submission/player.py starts with:', player_content[:80])
os.unlink(zip_path)
"
```
Expected: zip contains `aipoker/submission/player.py` with SpyAgent content.

- [ ] **Step 3: Verify leaderboard fetch works**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -c "
from auto_playwright import _fetch_leaderboard
teams = _fetch_leaderboard()
print(f'Teams fetched: {len(teams)}')
geoz = next((t for t in teams if t[\"name\"] == \"geoz\"), None)
if geoz:
    print(f'geoz: rank #{geoz[\"rank\"]}, ELO {geoz[\"elo\"]}, matches {geoz[\"matches\"]}')
else:
    print('geoz not found in leaderboard')
assert len(teams) > 0, 'Leaderboard fetch returned no teams'
"
```
Expected: Prints 30–60 teams, including geoz's current stats.

- [ ] **Step 4: Run all tests one final time**

```bash
cd /home/g30rgez/poker/poker-engine-2026
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: info-extraction pipeline complete — spy bot, playwright daemon, profiler"
```

---

## Deferred Metrics (post-first-run)

These spec metrics are intentionally excluded from this plan to meet the 6-day deadline. Add them after the first overnight run if the primary FTR/PF data proves insufficient:

- `avg_bet_frac_{street}` — opponent average bet size relative to pot, by street
- `sd_hand_dist` — showdown hand type distribution (pair/flush/trips/etc.)
- `adapts` — first-half vs. second-half FTR delta (requires same-mode hand inference)

---

## Overnight Usage Cheatsheet

```bash
# Before bed:
python auto_playwright.py --setup          # one-time: login + save session + validate submit selectors
python auto_playwright.py --submit spy     # swap in spy bot (takes ~60s)
python auto_playwright.py --daemon --interval 15 --bot spy  # start overnight loop

# Morning:
# Ctrl+C to stop daemon
python auto_playwright.py --submit main    # swap back production bot
python opponent_profiler.py --top 20       # read exploit report
```
