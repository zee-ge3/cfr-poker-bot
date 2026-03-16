"""
Continuous round-robin match runner.

Auto-discovers all Agent subclasses in submission/*.py and runs
an infinite round-robin (all pairs), cycling through matchups and
printing a live leaderboard after every match.

Usage:
  python run.py                  # discovers all bots, 1000 hands/match
  python run.py --hands 200      # faster iteration
  python run.py --hands 1000 --once   # one full round-robin then exit
"""
import argparse
import importlib
import inspect
import logging
import multiprocessing
import os
import signal
import sys
import time
from itertools import combinations, cycle

from agents.agent import Agent
import match as _match_module
from match import run_api_match

BASE_PORT = 9500
SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "submission")


# ── Bot discovery ─────────────────────────────────────────────────────────────

def discover_bots() -> dict[str, type]:
    """
    Scan submission/*.py for Agent subclasses.
    Returns {ClassName: class} sorted by filename for determinism.
    """
    bots = {}
    for fname in sorted(os.listdir(SUBMISSION_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_name = f"submission.{fname[:-3]}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  [WARN] Could not import {module_name}: {e}")
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Agent) and obj is not Agent and obj.__module__ == module_name:
                bots[name] = obj
    return bots


# ── Match runner ──────────────────────────────────────────────────────────────

def _kill_port(port):
    """Kill any process currently listening on port (clears orphaned servers)."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
                try:
                    psutil.Process(conn.pid).kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception:
        pass


def _kill_tree(pid):
    """Kill a process and all its descendants."""
    try:
        import psutil
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        proc.kill()
    except (psutil.NoSuchProcess, AttributeError):
        pass


def run_bot_process(cls, port, player_id):
    # Make this process a session leader so kill(-pgid) reaches uvicorn's child
    os.setsid()
    cls.run(True, port, player_id=player_id)


def run_one_match(name_a, cls_a, port_a, name_b, cls_b, port_b,
                  num_hands, match_idx, logger):
    """Start both bot processes, run match via HTTP, clean up."""
    # Clear any lingering servers from previous runs
    _kill_port(port_a)
    _kill_port(port_b)

    p0 = multiprocessing.Process(
        target=run_bot_process, args=(cls_a, port_a, "bot0"), daemon=True
    )
    p1 = multiprocessing.Process(
        target=run_bot_process, args=(cls_b, port_b, "bot1"), daemon=True
    )
    p0.start(); p1.start()
    time.sleep(0.5)   # give servers time to start

    # Reset match.py's module-level globals — none of them reset between calls
    _match_module.bankrolls = [0, 0]
    _match_module.time_used_0 = 0.0
    _match_module.time_used_1 = 0.0
    _match_module.failure_tracker = _match_module.AgentFailureTracker()

    csv_path = f"./match_{match_idx:04d}_{name_a}_vs_{name_b}.csv"
    try:
        result = run_api_match(
            f"http://localhost:{port_a}",
            f"http://localhost:{port_b}",
            logger,
            num_hands=num_hands,
            csv_path=csv_path,
            team_0_name=name_a,
            team_1_name=name_b,
        )
    except Exception as e:
        logger.error(f"Match error: {e}")
        result = None
    finally:
        # Kill full process trees (uvicorn spawns a grandchild server process
        # that survives p0.terminate() without this)
        _kill_tree(p0.pid)
        _kill_tree(p1.pid)
        p0.join(timeout=2)
        p1.join(timeout=2)

    return result, csv_path


# ── Leaderboard display ───────────────────────────────────────────────────────

def print_leaderboard(stats: dict, hands_per_match: int):
    """Print ranked leaderboard from accumulated stats."""
    rows = []
    for name, s in stats.items():
        m = s["matches"]
        if m == 0:
            continue
        ev = s["chips"] / (m * hands_per_match) if m else 0.0
        rows.append((name, ev, m, s["wins"], s["chips"]))
    rows.sort(key=lambda r: r[1], reverse=True)

    W = max((len(r[0]) for r in rows), default=12) + 2
    print(f"\n{'─'*60}")
    print(f"  {'#':<4} {'Bot':<{W}} {'EV/h':>7}  {'W-L':>8}  {'Net':>8}")
    print(f"  {'─'*54}")
    for i, (name, ev, m, wins, chips) in enumerate(rows, 1):
        wl = f"{wins}-{m-wins}"
        print(f"  {i:<4} {name:<{W}} {ev:>+7.2f}  {wl:>8}  {chips:>+8.0f}")
    print(f"{'─'*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--once",  action="store_true",
                        help="Run one full round-robin then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("runner")

    bots = discover_bots()
    if len(bots) < 2:
        print(f"Need at least 2 bots in {SUBMISSION_DIR}, found: {list(bots)}")
        sys.exit(1)

    names = list(bots.keys())
    print(f"\n{'='*60}")
    print(f"  Discovered {len(bots)} bots: {names}")
    print(f"  {len(list(combinations(names, 2)))} matchups per round  |  {args.hands} hands/match")
    print(f"{'='*60}\n")

    # Assign a fixed port to each bot
    ports = {name: BASE_PORT + i for i, name in enumerate(names)}

    # Per-bot cumulative stats
    stats = {name: {"chips": 0.0, "matches": 0, "wins": 0} for name in names}

    # Build ordered matchup list: all pairs, then repeat
    all_pairs = list(combinations(names, 2))
    matchup_gen = cycle(all_pairs)
    match_idx = 0

    try:
        for name_a, name_b in matchup_gen:
            cls_a, cls_b = bots[name_a], bots[name_b]
            port_a, port_b = ports[name_a], ports[name_b]
            match_idx += 1

            print(f"  Match {match_idx:4d}: {name_a} vs {name_b}  ({args.hands} hands)")

            result, csv_path = run_one_match(
                name_a, cls_a, port_a,
                name_b, cls_b, port_b,
                args.hands, match_idx, logger,
            )

            if result is not None:
                # Always read from CSV — match.py's global bankrolls accumulate
                # across calls and can't be trusted for per-match results.
                net_a = net_b = 0.0
                try:
                    import csv as _csv
                    with open(csv_path) as f:
                        rows = [r for r in _csv.DictReader(
                            [l for l in f if not l.startswith("#")]
                        )]
                    if rows:
                        net_a = float(rows[-1].get("team_0_bankroll", 0))
                        net_b = float(rows[-1].get("team_1_bankroll", 0))
                except Exception as csv_err:
                    logger.warning(f"CSV parse failed: {csv_err}")

                stats[name_a]["chips"]   += net_a
                stats[name_b]["chips"]   += net_b
                stats[name_a]["matches"] += 1
                stats[name_b]["matches"] += 1
                if net_a > net_b:
                    stats[name_a]["wins"] += 1
                elif net_b > net_a:
                    stats[name_b]["wins"] += 1

                ev_a = net_a / args.hands
                ev_b = net_b / args.hands
                winner = name_a if net_a > net_b else name_b
                print(f"    → {name_a}: {net_a:+.0f} ({ev_a:+.2f}/h)   "
                      f"{name_b}: {net_b:+.0f} ({ev_b:+.2f}/h)   "
                      f"Winner: {winner}")
            else:
                print(f"    → Match failed, skipping")

            # Print leaderboard after every complete round-robin
            if match_idx % len(all_pairs) == 0:
                print(f"\n  ══ Round {match_idx // len(all_pairs)} complete ══")
                print_leaderboard(stats, args.hands)

            if args.once and match_idx >= len(all_pairs):
                break

    except KeyboardInterrupt:
        print("\n\nStopped.")
        print_leaderboard(stats, args.hands)


if __name__ == "__main__":
    main()
