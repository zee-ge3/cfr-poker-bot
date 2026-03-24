"""
Benchmark the CFR-D bot against Slumbot (slumbot.com).

Slumbot plays HU NLHE with 50/100 blinds and 20,000 starting stacks (200bb).
Communication is via HTTP POST to https://slumbot.com/api/.

Usage:
  python slumbot_benchmark.py [--hands 100]
"""

import argparse
import json
import sys
import time
import requests
import numpy as np

from nlhe.cfr.preflop import PreflopTable
from nlhe.cfr.equity import card_str_to_idx
from nlhe.cfr.abstraction import (
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, RAISE_OVERBET,
    resolve_raise_amount,
)
from nlhe.bot import Bot
from nlhe.game import GameState

API_BASE = "https://slumbot.com/api"
STARTING_STACK = 20000
SB_BLIND = 50
BB_BLIND = 100

_preflop_table = PreflopTable()
_action_map = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN']
_bot = Bot(budget_seconds=3.0)  # 3 seconds per decision


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action_string(action_str: str):
    """
    Parse Slumbot's action string into a list of (street, actions) tuples.
    Streets are separated by '/'. Actions within a street are concatenated.

    Returns list of per-street action lists, where each action is:
      ('k',)       -- check
      ('c',)       -- call
      ('f',)       -- fold
      ('b', amount) -- bet/raise to `amount` total on this street
    """
    streets = action_str.split('/')
    result = []
    for street_str in streets:
        actions = []
        i = 0
        while i < len(street_str):
            ch = street_str[i]
            if ch == 'k':
                actions.append(('k',))
                i += 1
            elif ch == 'c':
                actions.append(('c',))
                i += 1
            elif ch == 'f':
                actions.append(('f',))
                i += 1
            elif ch == 'b':
                # Read the number following 'b'
                j = i + 1
                while j < len(street_str) and street_str[j].isdigit():
                    j += 1
                amount = int(street_str[i+1:j])
                actions.append(('b', amount))
                i = j
            else:
                i += 1
        result.append(actions)
    return result


def compute_game_state(action_str: str, hole_cards: list, board: list,
                       client_pos: int):
    """
    From Slumbot's action string, compute the current game state:
      - street (0=preflop, 1=flop, 2=turn, 3=river)
      - pot
      - our stack remaining
      - opponent's stack remaining
      - current bet to call
      - whose turn it is

    client_pos: 0 = we are BB, 1 = we are SB/BTN
    """
    streets = parse_action_string(action_str)

    # Track stacks. Player 0 = SB, Player 1 = BB
    # client_pos 0 = we are BB = player 1
    # client_pos 1 = we are SB = player 0
    stacks = [STARTING_STACK, STARTING_STACK]
    pot = 0

    # Post blinds
    stacks[0] -= SB_BLIND  # SB
    stacks[1] -= BB_BLIND  # BB
    pot = SB_BLIND + BB_BLIND

    street_idx = 0
    current_bet = [0, 0]  # per-player bet on current street

    # Preflop: SB posted 50, BB posted 100
    current_bet = [SB_BLIND, BB_BLIND]

    # Preflop: SB acts first
    actor = 0  # player index (0=SB, 1=BB)

    for s_idx, street_actions in enumerate(streets):
        if s_idx > 0:
            # New street: reset bets, BB acts first postflop
            current_bet = [0, 0]
            actor = 1  # BB acts first postflop (player 1)
            street_idx = s_idx

        for act in street_actions:
            if act[0] == 'k':
                actor = 1 - actor
            elif act[0] == 'c':
                # Call: match the other player's bet
                to_call = max(current_bet[1 - actor] - current_bet[actor], 0)
                actual_call = min(to_call, stacks[actor])
                stacks[actor] -= actual_call
                current_bet[actor] += actual_call
                pot += actual_call
                actor = 1 - actor
            elif act[0] == 'f':
                pass  # hand over
            elif act[0] == 'b':
                # Bet to X total on this street
                bet_to = act[1]
                additional = bet_to - current_bet[actor]
                additional = min(additional, stacks[actor])
                stacks[actor] -= additional
                current_bet[actor] += additional
                pot += additional
                actor = 1 - actor

    # Whose turn is it? The `actor` variable now points to next-to-act
    # Map to our perspective
    if client_pos == 0:
        # We are BB = player 1
        our_stack = stacks[1]
        opp_stack = stacks[0]
        our_bet = current_bet[1]
        opp_bet = current_bet[0]
        our_position = 1  # BB
        its_our_turn = (actor == 1)
    else:
        # We are SB = player 0
        our_stack = stacks[0]
        opp_stack = stacks[1]
        our_bet = current_bet[0]
        opp_bet = current_bet[1]
        our_position = 0  # SB
        its_our_turn = (actor == 0)

    to_call = max(opp_bet - our_bet, 0)

    return {
        'street': street_idx,
        'pot': pot,
        'our_stack': our_stack,
        'opp_stack': opp_stack,
        'to_call': to_call,
        'our_bet_this_street': our_bet,
        'opp_bet_this_street': opp_bet,
        'our_position': our_position,
        'its_our_turn': its_our_turn,
        'hole_cards': hole_cards,
        'board': board,
    }


# ---------------------------------------------------------------------------
# Bot decision
# ---------------------------------------------------------------------------

def bot_decide(state: dict, use_solver: bool = False) -> str:
    """
    Given game state, return Slumbot action string (k, c, f, or bX).

    If use_solver=True, uses the full Bot class with postflop CFR-D solver.
    Otherwise uses preflop table + check/call postflop.
    """
    if use_solver:
        return _decide_with_solver(state)

    if state['street'] == 0:
        return _decide_preflop(state)
    else:
        return _decide_postflop(state)


def _notify_opp_actions(prev_action_str: str, new_action_str: str, client_pos: int):
    """Parse new actions since prev and notify bot of opponent actions."""
    # Find the new suffix
    if new_action_str.startswith(prev_action_str):
        new_part = new_action_str[len(prev_action_str):]
    else:
        return  # action strings diverged (street boundary), skip

    # Parse new actions (alternating actors)
    # Determine whose turn it was at end of prev_action_str
    prev_streets = parse_action_string(prev_action_str) if prev_action_str else []
    # Count actions in the last street to determine actor
    # For simplicity, just notify for any bet/raise/call/fold in new_part
    # Map client_pos: 0=BB, 1=SB
    i = 0
    while i < len(new_part):
        ch = new_part[i]
        if ch == 'b':
            j = i + 1
            while j < len(new_part) and new_part[j].isdigit():
                j += 1
            _bot.observe_action('RAISE_LARGE')
            i = j
        elif ch == 'c':
            _bot.observe_action('CALL')
            i += 1
        elif ch == 'k':
            _bot.observe_action('CALL')  # check
            i += 1
        elif ch == 'f':
            _bot.observe_action('FOLD')
            i += 1
        elif ch == '/':
            # Street boundary: clear stale solver so subsequent opponent
            # actions on the new street don't use old tree for narrowing.
            _bot._solver = None
            i += 1
        else:
            i += 1


def _decide_with_solver(state: dict) -> str:
    """Use the full Bot class for decision making."""
    to_call = state['to_call']
    our_stack = state['our_stack']
    pot = state['pot']

    # Build valid actions -- standard action abstraction.
    # RAISE_ALLIN only when SPR < 5 (short enough that shoving is part of
    # a balanced strategy). At higher SPR the tree uses sized bets only.
    valid = []
    if to_call > 0:
        valid.append('FOLD')
    valid.append('CALL')

    effective_stack = min(our_stack, state['opp_stack'])
    spr = effective_stack / max(pot, 1)

    if our_stack > 0 and our_stack > to_call:
        # Match solver's sizing: resolve_raise_amount(action, pot+call, remaining)
        pot_after_call = pot + to_call
        remaining = max(0, our_stack - to_call)
        small = int(resolve_raise_amount(RAISE_SMALL, pot_after_call, remaining))
        large = int(resolve_raise_amount(RAISE_LARGE, pot_after_call, remaining))
        overbet = int(resolve_raise_amount(RAISE_OVERBET, pot_after_call, remaining))
        allin = remaining
        if small < allin:
            valid.append('RAISE_SMALL')
        if large < allin and large != small:
            valid.append('RAISE_LARGE')
        if overbet < allin and overbet not in (small, large):
            valid.append('RAISE_OVERBET')
        if spr < 5:
            valid.append('RAISE_ALLIN')

    gs = GameState(
        street=state['street'],
        pot=pot,
        our_stack=our_stack,
        opp_stack=state['opp_stack'],
        our_hole=state['hole_cards'],
        opp_hole=[],
        board=state['board'],
        valid_actions=valid,
        position=state['our_position'],
        street_bets=[],
        to_call=to_call,
    )

    try:
        action = _bot.decide(gs)
        return _action_to_slumbot(action, state)
    except Exception as e:
        # Fallback to heuristic
        return _decide_postflop(state) if state['street'] > 0 else _decide_preflop(state)


def _decide_preflop(state: dict) -> str:
    hole = state['hole_cards']
    position = state['our_position']
    to_call = state['to_call']
    our_stack = state['our_stack']
    pot = state['pot']
    our_bet = state['our_bet_this_street']

    try:
        idxs = tuple(sorted(card_str_to_idx(c) for c in hole))
        probs = np.array(_preflop_table.lookup(idxs, position), dtype=float)

        # Build valid actions
        valid = []
        if to_call > 0:
            valid.append('FOLD')
        valid.append('CALL')
        if our_stack > 0:
            valid.append('RAISE_ALLIN')
            if our_stack > to_call:
                valid.extend(['RAISE_SMALL', 'RAISE_LARGE'])

        weights = np.zeros(len(valid))
        for i, a in enumerate(valid):
            if a in _action_map:
                weights[i] = probs[_action_map.index(a)]
        if weights.sum() < 1e-10:
            weights = np.ones(len(valid))
        weights /= weights.sum()

        chosen = valid[np.random.choice(len(valid), p=weights)]

        return _action_to_slumbot(chosen, state)
    except Exception as e:
        # Fallback: check or call
        if to_call > 0:
            return 'c'
        return 'k'


def _decide_postflop(state: dict) -> str:
    """
    Postflop heuristic using MC equity.

    Strategy:
    - Value bet strongly with equity > 0.70
    - Thin value bet with equity > 0.55 (only as first bettor)
    - Check-call medium hands, fold weak
    - Fold to large bets unless we have strong equity
    - Include some bluffs with low-equity hands that have blockers
    """
    from nlhe.cfr.equity import equity_mc
    to_call = state['to_call']
    pot = state['pot']
    our_stack = state['our_stack']
    opp_stack = state['opp_stack']
    hole = state['hole_cards']
    board = state['board']
    our_bet = state['our_bet_this_street']
    street = state['street']

    if not board:
        if to_call > 0:
            return 'c' if to_call <= pot * 0.35 else 'f'
        return 'k'

    try:
        n_samples = 400 if len(board) < 5 else 600
        eq = equity_mc(hole, board, n_samples=n_samples)
    except Exception:
        eq = 0.5

    effective_stack = min(our_stack, opp_stack)
    spr = effective_stack / max(pot, 1)

    if to_call > 0:
        # Facing a bet
        pot_odds = to_call / (pot + to_call)
        bet_frac = to_call / max(pot - to_call, 1)

        # Against large bets (> 75% pot), require more equity
        if bet_frac > 1.0:
            # Overbet: only call with strong hands
            if eq > 0.70:
                return 'c'
            return 'f'
        elif bet_frac > 0.6:
            # Large bet: need decent equity
            if eq > 0.85 and spr > 1:
                # Nuts: raise
                raise_size = int(to_call + pot * 0.75)
                raise_to = our_bet + min(to_call + raise_size, our_stack)
                opp_bet = state['opp_bet_this_street']
                raise_to = max(raise_to, min(opp_bet * 2, our_bet + our_stack))
                return f'b{raise_to}'
            elif eq > pot_odds + 0.10:
                return 'c'
            else:
                return 'f'
        else:
            # Small/medium bet
            if eq > 0.80 and spr > 1.5:
                raise_size = int(to_call + pot * 0.7)
                raise_to = our_bet + min(to_call + raise_size, our_stack)
                opp_bet = state['opp_bet_this_street']
                raise_to = max(raise_to, min(opp_bet * 2, our_bet + our_stack))
                return f'b{raise_to}'
            elif eq > pot_odds + 0.05:
                return 'c'
            else:
                return 'f'
    else:
        # First to act or checked to
        if eq > 0.72:
            # Strong value bet: 55% pot
            bet_size = int(pot * 0.55)
            bet_size = max(bet_size, BB_BLIND)
            bet_size = min(bet_size, our_stack)
            return f'b{our_bet + bet_size}'
        elif eq > 0.58 and spr > 2:
            # Thin value: 33% pot
            bet_size = int(pot * 0.33)
            bet_size = max(bet_size, BB_BLIND)
            bet_size = min(bet_size, our_stack)
            return f'b{our_bet + bet_size}'
        elif eq < 0.20 and spr > 3 and street < 3 and np.random.random() < 0.25:
            # Bluff with very weak hands ~25% of the time on flop/turn
            bet_size = int(pot * 0.5)
            bet_size = max(bet_size, BB_BLIND)
            bet_size = min(bet_size, our_stack)
            return f'b{our_bet + bet_size}'
        else:
            return 'k'


def _action_to_slumbot(action: str, state: dict) -> str:
    """Convert our action string to Slumbot format.

    Uses resolve_raise_amount with solver-consistent parameters so the
    executed bet size matches what the CFR tree optimized for.
    """
    if action == 'FOLD':
        return 'f'
    if action == 'CALL':
        if state['to_call'] > 0:
            return 'c'
        return 'k'  # check

    # Raise actions: compute bet-to amount
    our_stack = state['our_stack']
    our_bet = state['our_bet_this_street']
    opp_bet = state['opp_bet_this_street']
    pot = state['pot']
    to_call = state['to_call']

    if action == 'RAISE_ALLIN':
        bet_to = our_bet + our_stack
    else:
        action_const = {
            'RAISE_SMALL': RAISE_SMALL,
            'RAISE_LARGE': RAISE_LARGE,
            'RAISE_OVERBET': RAISE_OVERBET,
        }[action]

        # Match the solver tree's sizing exactly:
        # solver calls resolve_raise_amount(action, pot+call, remaining_after_call)
        # with default current_bet=0.0
        pot_after_call = pot + to_call
        remaining = max(0, our_stack - to_call)
        raise_size = int(resolve_raise_amount(action_const, pot_after_call, remaining))
        raise_size = max(raise_size, BB_BLIND)  # minimum raise
        raise_size = min(raise_size, remaining)

        bet_to = our_bet + to_call + raise_size
        bet_to = min(bet_to, our_bet + our_stack)

        # Ensure minimum legal raise
        min_raise_to = opp_bet * 2 if opp_bet > 0 else BB_BLIND * 2
        bet_to = max(bet_to, min(min_raise_to, our_bet + our_stack))

    # If computed bet_to doesn't exceed opponent's bet, it's not a valid
    # raise -- fall back to call (or check).
    if bet_to <= opp_bet:
        return 'c' if to_call > 0 else 'k'

    return f'b{int(bet_to)}'


# ---------------------------------------------------------------------------
# Slumbot API client
# ---------------------------------------------------------------------------

def new_hand(token: str = None) -> dict:
    body = {}
    if token:
        body['token'] = token
    resp = requests.post(f"{API_BASE}/new_hand", json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def act(token: str, incr: str) -> dict:
    body = {'token': token, 'incr': incr}
    resp = requests.post(f"{API_BASE}/act", json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def play_hand(token: str = None, use_solver: bool = False) -> tuple:
    """
    Play one hand against Slumbot.
    Returns (winnings, new_token, hand_log).
    """
    resp = new_hand(token)
    token = resp.get('token', token)
    client_pos = resp['client_pos']
    hole_cards = resp['hole_cards']
    board = resp.get('board', [])
    action_str = resp.get('action', '')

    # Initialize the full bot for this hand
    if use_solver:
        position = 1 if client_pos == 0 else 0  # client_pos 0 = BB = position 1
        _bot.new_hand(hole_cards, position)

    log_lines = [f"  pos={'BB' if client_pos==0 else 'SB'} hole={hole_cards}"]

    # Check if hand is already over (e.g., we're BB and SB folded preflop)
    if 'winnings' in resp:
        log_lines.append(f"  result: {resp['winnings']:+d} (instant)")
        return resp['winnings'], token, log_lines

    prev_action_str = action_str or ''
    for step in range(50):  # safety limit
        board = resp.get('board', board or [])
        action_str = resp.get('action', action_str)

        # Notify bot of opponent actions since last decision
        if use_solver and action_str != prev_action_str:
            _notify_opp_actions(prev_action_str, action_str, client_pos)

        state = compute_game_state(action_str, hole_cards, board, client_pos)

        if not state['its_our_turn']:
            # Shouldn't happen if API is working correctly
            log_lines.append(f"  WARNING: not our turn? action={action_str}")
            break

        our_action = bot_decide(state, use_solver=use_solver)
        log_lines.append(f"  street={state['street']} pot={state['pot']} "
                        f"to_call={state['to_call']} -> {our_action}")

        # Track action string after our action for next diff
        prev_action_str = action_str + our_action

        resp = act(token, our_action)
        token = resp.get('token', token)

        if 'winnings' in resp:
            w = resp['winnings']
            board_final = resp.get('board', board)
            bot_hole = resp.get('bot_hole_cards', [])
            log_lines.append(f"  board={board_final} slumbot={bot_hole} "
                           f"result: {w:+d}")
            return w, token, log_lines

    log_lines.append("  WARNING: hand did not complete in 50 steps")
    return 0, token, log_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=100,
                       help='Number of hands to play')
    parser.add_argument('--solver', action='store_true',
                       help='Use full postflop solver (slower but stronger)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    n_hands = args.hands
    mode = "full solver" if args.solver else "preflop-only"
    print(f"Playing {n_hands} hands against Slumbot ({mode})...\n")

    token = None
    total_winnings = 0
    hand_results = []
    errors = 0

    for i in range(n_hands):
        try:
            w, token, log = play_hand(token, use_solver=args.solver)
            total_winnings += w
            hand_results.append(w)

            if args.verbose or abs(w) >= 2000:
                print(f"Hand {i+1} [{'+' if w >= 0 else ''}{w}]:")
                for line in log:
                    print(line)

            if (i + 1) % 10 == 0 or i == 0:
                avg = total_winnings / (i + 1)
                avg_bb = avg / BB_BLIND
                print(f"  Hand {i+1:4d}/{n_hands}: "
                      f"this={w:+6d}  total={total_winnings:+8d}  "
                      f"avg={avg:+.0f} chips/hand ({avg_bb:+.1f} bb/hand)")

            # Small delay to be respectful
            time.sleep(0.3)

        except requests.RequestException as e:
            errors += 1
            print(f"  Hand {i+1}: HTTP error: {e}")
            if errors > 10:
                print("Too many errors, stopping.")
                break
            time.sleep(2)
            token = None  # reset session
        except Exception as e:
            errors += 1
            print(f"  Hand {i+1}: Error: {e}")
            if errors > 10:
                print("Too many errors, stopping.")
                break

    n_played = len(hand_results)
    if n_played == 0:
        print("No hands completed.")
        return

    results = np.array(hand_results)
    avg = results.mean()
    std = results.std()
    avg_bb = avg / BB_BLIND

    print(f"\n{'='*50}")
    print(f"Results: {n_played} hands against Slumbot")
    print(f"{'='*50}")
    print(f"Total winnings:  {total_winnings:+d} chips")
    print(f"Avg per hand:    {avg:+.0f} chips ({avg_bb:+.2f} bb/hand)")
    print(f"Std dev:         {std:.0f} chips per hand")
    print(f"95% CI:          {avg - 1.96*std/np.sqrt(n_played):+.0f} to "
          f"{avg + 1.96*std/np.sqrt(n_played):+.0f}")
    print(f"Errors:          {errors}")

    # Context: a strong bot wins ~5-15 mbb/hand vs Slumbot
    # A decent bot: 0 to -50 mbb/hand
    # A bad bot: < -100 mbb/hand
    mbb = avg_bb * 1000
    print(f"\nPerformance:     {mbb:+.0f} mbb/hand")
    if mbb > 0:
        print("  (Positive! Beating Slumbot)")
    elif mbb > -50:
        print("  (Competitive -- within reasonable range)")
    elif mbb > -200:
        print("  (Below average but functional)")
    else:
        print("  (Significant losses -- needs work)")


if __name__ == '__main__':
    main()
