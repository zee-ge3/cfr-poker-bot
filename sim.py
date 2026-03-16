"""
Fast direct simulation — no HTTP, no subprocess.
Runs round-robin between all registered bots.

Usage:
  python sim.py                        # all bots, 3 trials x 300 hands each pair
  python sim.py --hands 500 --trials 5
  python sim.py --bots PlayerAgent Claude2Agent   # specific subset
  python sim.py --quiet                # suppress per-hand progress
"""
import sys, time, random, argparse
from itertools import combinations
from gym_env import PokerEnv


def run_match(agent0, agent1, num_hands=300, seed=None, quiet=False):
    if seed is not None:
        random.seed(seed)

    bankrolls = [0, 0]
    invalid_actions = [0, 0]
    raises = [0, 0]
    calls = [0, 0]
    folds = [0, 0]

    for hand_num in range(num_hands):
        env = PokerEnv()
        small_blind = hand_num % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind})
        info["hand_number"] = hand_num

        for obs in (obs0, obs1):
            obs["time_used"] = 0.0
            obs["time_left"] = 9999.0
            obs["opp_last_action"] = "None"

        reward0 = reward1 = 0
        terminated = truncated = False
        last_action = [None, None]

        while not terminated:
            acting = obs0["acting_agent"]
            if acting == 0:
                obs0["opp_last_action"] = "None" if last_action[1] is None else last_action[1]
                act = agent0.act(obs0, reward0, terminated, truncated, info)
            else:
                obs1["opp_last_action"] = "None" if last_action[0] is None else last_action[0]
                act = agent1.act(obs1, reward1, terminated, truncated, info)

            act_type = act[0]
            if act_type == 1:   raises[acting] += 1
            elif act_type == 3: calls[acting] += 1
            elif act_type == 0: folds[acting] += 1

            (obs0, obs1), (reward0, reward1), terminated, truncated, step_info = env.step(act)
            info.update(step_info)
            info["hand_number"] = hand_num

            for obs in (obs0, obs1):
                obs["time_used"] = 0.0
                obs["time_left"] = 9999.0

            act_name = PokerEnv.ActionType(act_type).name
            if step_info.get("invalid_action"):
                invalid_actions[acting] += 1
                act_name = "FOLD(invalid)"

            last_action[acting] = act_name

        agent0.observe(obs0, reward0, True, truncated, info)
        agent1.observe(obs1, reward1, True, truncated, info)
        agent0.act(obs0, reward0, True, truncated, info)
        agent1.act(obs1, reward1, True, truncated, info)

        bankrolls[0] += reward0
        bankrolls[1] += reward1

        if not quiet and (hand_num + 1) % 100 == 0:
            print(f"    Hand {hand_num+1:4d} | P0={bankrolls[0]:+6d}  P1={bankrolls[1]:+6d} | "
                  f"raises={raises}  calls={calls}", flush=True)

    return bankrolls, invalid_actions


def run_roundrobin(bot_classes: dict, hands: int, trials: int, seed=None, quiet=False):
    """
    Run round-robin between all bots.
    Each pair plays 2*trials matches (both sides) for fairness.
    Returns per-bot net EV/hand and per-matchup results.
    """
    names = list(bot_classes.keys())
    total_hands_per_pair = 2 * trials * hands

    # ev_matrix[a][b] = EV/hand for bot a when playing against bot b
    ev_matrix = {n: {} for n in names}
    net_ev = {n: 0.0 for n in names}
    matchup_count = {n: 0 for n in names}

    for name_a, name_b in combinations(names, 2):
        cls_a, cls_b = bot_classes[name_a], bot_classes[name_b]
        a_chips = 0
        b_chips = 0

        for trial in range(trials):
            trial_seed = (seed + trial) if seed is not None else None
            swap_seed  = (seed + trial + 10000) if seed is not None else None

            if not quiet:
                print(f"  [{name_a} vs {name_b}] trial {trial+1}/{trials} (normal):", flush=True)
            a0 = cls_a(stream=False)
            a1 = cls_b(stream=False)
            br, inv = run_match(a0, a1, num_hands=hands, seed=trial_seed, quiet=quiet)
            a_chips += br[0]
            b_chips += br[1]

            if not quiet:
                print(f"  [{name_b} vs {name_a}] trial {trial+1}/{trials} (swapped):", flush=True)
            a0 = cls_b(stream=False)
            a1 = cls_a(stream=False)
            br, inv = run_match(a0, a1, num_hands=hands, seed=swap_seed, quiet=quiet)
            a_chips += br[1]   # a was P1 this time
            b_chips += br[0]

        ev_a = a_chips / total_hands_per_pair
        ev_b = b_chips / total_hands_per_pair
        ev_matrix[name_a][name_b] = ev_a
        ev_matrix[name_b][name_a] = ev_b
        net_ev[name_a] += ev_a
        net_ev[name_b] += ev_b
        matchup_count[name_a] += 1
        matchup_count[name_b] += 1

    return ev_matrix, net_ev, matchup_count


def print_results(ev_matrix, net_ev, matchup_count, hands, trials):
    names = list(net_ev.keys())
    ranked = sorted(names, key=lambda n: net_ev[n], reverse=True)
    total_hands_per_pair = 2 * trials * hands

    W = max(len(n) for n in names) + 2

    print(f"\n{'='*70}")
    print(f"  ROUND-ROBIN RESULTS  ({trials} trials × {hands} hands each side = {total_hands_per_pair} hands/pair)")
    print(f"{'='*70}")

    # Head-to-head matrix
    print(f"\n  EV/hand matrix (row = bot, col = opponent):")
    header = f"  {'':>{W}}" + "".join(f"  {n:>{W}}" for n in ranked)
    print(header)
    for name in ranked:
        row = f"  {name:>{W}}"
        for opp in ranked:
            if opp == name:
                row += f"  {'---':>{W}}"
            else:
                ev = ev_matrix[name].get(opp, 0.0)
                row += f"  {ev:>+{W}.2f}"
        print(row)

    # Rankings
    print(f"\n  RANKINGS (net EV/hand summed across all opponents):")
    print(f"  {'#':<4} {'Bot':<{W+2}} {'Net EV/h':>10}  {'Matchups':>8}")
    print(f"  {'-'*40}")
    for i, name in enumerate(ranked, 1):
        avg = net_ev[name] / matchup_count[name] if matchup_count[name] else 0
        print(f"  {i:<4} {name:<{W+2}} {net_ev[name]:>+10.2f}  ({matchup_count[name]} matchups, avg {avg:+.2f}/h)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands",  type=int,   default=300)
    parser.add_argument("--trials", type=int,   default=3)
    parser.add_argument("--seed",   type=int,   default=None)
    parser.add_argument("--quiet",  action="store_true", help="Suppress per-hand output")
    parser.add_argument("--bots",   nargs="+",  default=None,
                        help="Subset of bots to run (e.g. --bots PlayerAgent Claude2Agent)")
    args = parser.parse_args()

    # ── Bot registry ──────────────────────────────────────────────────────────
    from submission.player import PlayerAgent
    from submission.claude2_player import Claude2Agent

    ALL_BOTS = {
        "PlayerAgent":  PlayerAgent,
        "Claude2Agent": Claude2Agent,
    }

    # Optionally register a third bot here when ready:
    # from submission.ensemble_player import EnsembleAgent
    # ALL_BOTS["EnsembleAgent"] = EnsembleAgent

    if args.bots:
        missing = [b for b in args.bots if b not in ALL_BOTS]
        if missing:
            print(f"Unknown bots: {missing}. Available: {list(ALL_BOTS)}")
            sys.exit(1)
        bot_classes = {b: ALL_BOTS[b] for b in args.bots}
    else:
        bot_classes = ALL_BOTS

    if len(bot_classes) < 2:
        print("Need at least 2 bots for a round-robin.")
        sys.exit(1)

    print(f"Bots: {list(bot_classes)}")
    print(f"Config: {args.trials} trials × {args.hands} hands/side")
    print(f"Total hands per pair: {2 * args.trials * args.hands}\n")

    t0 = time.time()
    ev_matrix, net_ev, matchup_count = run_roundrobin(
        bot_classes, args.hands, args.trials, seed=args.seed, quiet=args.quiet
    )
    elapsed = time.time() - t0

    print_results(ev_matrix, net_ev, matchup_count, args.hands, args.trials)
    total_hands = sum(matchup_count.values()) // 2 * 2 * args.trials * args.hands
    print(f"  Time: {elapsed:.1f}s  ({elapsed/total_hands*1000:.0f}ms/hand)")
