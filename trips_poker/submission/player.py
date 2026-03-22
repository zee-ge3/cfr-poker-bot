"""
Poker Bot v7 — PlayerAgent orchestrator.
Trips-Aware Preflop + Global RHS + SQL Laboratory.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import sys
import math
import random
import time
from collections import deque

import numpy as np

# Ensure submission package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from submission.equity import exact_equity, optimal_discard, preflop_strength, hand_rank
from submission.opponent import OpponentModel
from submission.strategy import decide
from submission.match_manager import MatchManager
from submission.card_utils import classify_hand, NUM_CARDS, combo_index_5
try:
    from submission.logger_sql import SQLMatchLogger
except ImportError:
    SQLMatchLogger = None
from submission.cfr.subgame_cfr import SubgameSolver
from submission.cfr.variance_control import select_action
from submission.cfr.action_abstraction import (
    abstract_action_from_real, resolve_raise_amount, N_ACTIONS, ALL_RAISES,
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, RAISE_OVERBET
)
# from submission.ppo_bridge import get_ppo_bridge  # V10.1: removed — never used

ACTION_NAMES = {0: 'FOLD', 1: 'RAISE', 2: 'CHECK', 3: 'CALL', 4: 'DISCARD'}

# ── Preflop CFR table support ────────────────────────────────────────────────
# Build mapping from preflop action histories to BFS-indexed node indices,
# matching PreflopCFR's _bfs_index_nodes() ordering exactly.
_PREFLOP_NODE_MAP = None  # lazy-init cache


def _build_preflop_node_map():
    """Build (ip_map, oop_map, facing_map) matching PreflopCFR BFS ordering.

    Returns:
        ip_map: {action_history_tuple: ip_node_idx}
        oop_map: {action_history_tuple: oop_node_idx}
        facing_map: {action_history_tuple: facing_amount} for raise classification
    """
    RAISE_CAP = 3
    STACK = 200.0

    ip_map, oop_map, facing_map = {}, {}, {}
    ip_count = oop_count = 0

    # BFS: (player, raises, is_root, history, ip_committed, oop_committed)
    queue = deque([(0, 0, True, (), 1.0, 2.0)])

    while queue:
        player, raises, is_root, history, ip_c, oop_c = queue.popleft()
        facing = max(ip_c, oop_c)
        facing_map[history] = facing

        if player == 0:
            ip_map[history] = ip_count
            ip_count += 1
        else:
            oop_map[history] = oop_count
            oop_count += 1

        # Valid actions (same logic as PreflopCFR._valid_actions)
        actions = [FOLD, CALL]
        if raises < RAISE_CAP:
            actions.extend([RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN])

        for a in sorted(actions):
            if a == FOLD:
                continue  # terminal
            if a == CALL:
                if is_root:
                    # SB limp → BB gets to act (non-terminal)
                    new_ip = max(ip_c, oop_c) if player == 0 else ip_c
                    new_oop = max(ip_c, oop_c) if player == 1 else oop_c
                    queue.append((1 - player, raises, False, history + (a,), new_ip, new_oop))
                # Non-root CALL is terminal — skip
                continue
            # Raise actions
            if a == RAISE_SMALL:
                total = max(facing * 3.0, 6.0) if raises == 0 else max(facing * 3.0, facing + 3.0)
            elif a == RAISE_LARGE:
                total = max(facing * 10.0, 16.0) if raises == 0 else max(facing * 10.0, facing + 10.0)
            else:
                total = STACK
            total = min(total, STACK)
            new_ip = total if player == 0 else ip_c
            new_oop = total if player == 1 else oop_c
            queue.append((1 - player, raises + 1, False, history + (a,), new_ip, new_oop))

    return ip_map, oop_map, facing_map


def _get_preflop_node_map():
    global _PREFLOP_NODE_MAP
    if _PREFLOP_NODE_MAP is None:
        _PREFLOP_NODE_MAP = _build_preflop_node_map()
    return _PREFLOP_NODE_MAP


def _classify_preflop_raise(raise_to: float, facing: float) -> int:
    """Map a real preflop raise-to amount to the nearest abstract action."""
    if raise_to >= 150:  # close to all-in (200 stack)
        return RAISE_ALLIN
    ratio = raise_to / max(1.0, facing)
    # Midpoint between RS (3×) and RL (10×) on log scale ≈ 5.5×
    if ratio < 5.5:
        return RAISE_SMALL
    return RAISE_LARGE


class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = False, player_id: str = None):
        super().__init__(stream=stream, player_id=player_id)
        self.opponent = OpponentModel()
        self.match_mgr = MatchManager()
        self.sql_logger = SQLMatchLogger() if SQLMatchLogger is not None else None
        # Cache evaluator for showdown processing
        from gym_env import PokerEnv
        self._evaluator = PokerEnv().evaluator
        self._int_card_to_str = PokerEnv.int_card_to_str

        # Per-hand state
        self._my_cards = []
        self._my_kept = None       # (card1, card2) after discard
        self._my_discards = None   # (card1, card2, card3) after discard
        self._board = []
        self._opp_discards = None
        self._opp_showdown_cards = None
        self._street = -1
        self._last_street = -1
        self._we_raised = False
        self._hands_played = 0
        # In-hand action history for Bayesian range narrowing
        # List of (action_str, street) tuples — opponent actions only
        self._hand_actions: list = []
        self._range_collapsed = False  # True when preflop narrowing yields 0 active hands

        # CFR solver with trained value networks for depth-limited evaluation
        self._solver = SubgameSolver()
        from submission.cfr.value_net import load_value_net
        _model_dir = os.path.join(os.path.dirname(__file__), 'tables')
        self._value_net_turn = load_value_net(os.path.join(_model_dir, 'value_net_turn.pt'))
        self._value_net_flop = load_value_net(os.path.join(_model_dir, 'value_net_flop.pt'))
        self._value_net_river = load_value_net(os.path.join(_model_dir, 'value_net_river.pt'))

        # V10: PPO bridge removed — was loaded but never called (dead code)

        # Preflop CFR tables (solved strategies indexed by hand × tree node)
        # Only load 'neutral' — protection/pressure variants are never used
        # (player.py always selects 'neutral') and waste 124MB of RAM.
        self._preflop_tables = {}
        for variant in ('neutral',):
            ip_p = os.path.join(_model_dir, f'preflop_strategy_ip_{variant}.npy')
            oop_p = os.path.join(_model_dir, f'preflop_strategy_oop_{variant}.npy')
            if os.path.exists(ip_p) and os.path.exists(oop_p):
                self._preflop_tables[variant] = {
                    'ip': np.load(ip_p),    # (80730, 40, 5)
                    'oop': np.load(oop_p),  # (80730, 40, 5)
                }
        self._preflop_history = ()  # track abstract actions within a preflop hand

    def _compute_preflop_likelihoods(self, in_position):
        """Compute per-5-card-hand likelihoods from opponent's preflop actions.

        Uses our preflop CFR tables in reverse: for each possible opponent
        5-card hand, compute P(they took the observed preflop actions | hand).
        Returns (80730,) array of likelihoods, or None if no narrowing possible.
        """
        if not self._preflop_tables or not self._preflop_history:
            return None

        ip_map, oop_map, _ = _get_preflop_node_map()

        # Use neutral variant (we don't know opponent's margin perception)
        variant = 'neutral'
        if variant not in self._preflop_tables:
            variant = next(iter(self._preflop_tables), None)
        if variant is None:
            return None

        # Opponent is the opposite position from us
        opp_is_ip = not in_position
        opp_table = self._preflop_tables[variant]['ip' if opp_is_ip else 'oop']
        opp_node_map = ip_map if opp_is_ip else oop_map

        n_hands = opp_table.shape[0]  # 80730 = C(27,5)
        likelihoods = np.ones(n_hands, dtype=np.float64)
        narrowed = False

        current_history = ()
        for action in self._preflop_history:
            # Determine which player is acting at this history prefix
            if current_history in ip_map:
                acting_is_ip = True
            elif current_history in oop_map:
                acting_is_ip = False
            else:
                break

            # Only narrow on opponent's actions
            if acting_is_ip == opp_is_ip:
                node_idx = opp_node_map.get(current_history)
                if node_idx is not None and node_idx < opp_table.shape[1] and action < opp_table.shape[2]:
                    action_probs = opp_table[:, node_idx, action].astype(np.float64)
                    # Floor at 0.01 to avoid zeroing out hands entirely
                    # (opponent may deviate from Nash)
                    action_probs = np.maximum(action_probs, 0.01)
                    likelihoods *= action_probs
                    narrowed = True

            current_history = current_history + (action,)

        # Also narrow on opponent's terminal CALL: if the last tracked action
        # was ours (a raise), the opponent must have called to reach postflop.
        if self._preflop_history:
            last_history = current_history  # = full preflop_history
            if last_history in ip_map:
                last_acting_is_ip = True
            elif last_history in oop_map:
                last_acting_is_ip = False
            else:
                last_acting_is_ip = None

            # If the opponent is acting next (it's their turn after our last raise)
            # and we reached postflop, they must have called
            if last_acting_is_ip is not None and last_acting_is_ip == opp_is_ip:
                node_idx = opp_node_map.get(last_history)
                if node_idx is not None and node_idx < opp_table.shape[1] and CALL < opp_table.shape[2]:
                    call_probs = opp_table[:, node_idx, CALL].astype(np.float64)
                    call_probs = np.maximum(call_probs, 0.01)
                    likelihoods *= call_probs
                    narrowed = True

        if not narrowed:
            return None

        return likelihoods

    def _card_str(self, cards):
        """Convert list of int cards to readable strings."""
        try:
            return [self._int_card_to_str(c) for c in cards if c >= 0]
        except Exception:
            return [str(c) for c in cards]

    def act(self, observation, reward, terminated, truncated, info) -> tuple:
        obs = observation

        # Terminal call: no-op
        if terminated:
            return (0, 0, 0, 0)

        # Parse observation
        street = obs['street']
        valid = obs['valid_actions']
        my_cards = [c for c in obs['my_cards'] if c != -1]
        community = [c for c in obs['community_cards'] if c != -1]
        opp_discards = [c for c in obs['opp_discarded_cards'] if c != -1]
        my_discards_obs = [c for c in obs['my_discarded_cards'] if c != -1]
        my_bet = obs['my_bet']
        opp_bet = obs['opp_bet']
        min_raise = obs['min_raise']
        max_raise = obs['max_raise']
        opp_last = obs.get('opp_last_action', 'None')
        in_position = (obs.get('blind_position', 0) == 0)  # 0=IP(SB/btn), 1=OOP(BB)

        # Track opponent actions
        if opp_last and opp_last != 'None' and opp_last != 'DISCARD':
            # Skip forced runout checks when both players are all-in (max_raise==0).
            # These are not real decisions; counting them taints the opponent model
            # and CFR regret sums with phantom actions at unreachable game states.
            is_all_in_runout = (max_raise == 0 and 'CHECK' in opp_last)
            if not is_all_in_runout:
                opp_raise_size = max(0, opp_bet - my_bet) if 'RAISE' in opp_last else 0
                pot = my_bet + opp_bet
                self.opponent.update_action(opp_last, street, opp_raise_size, pot)

                # Track fold/call response to our raises
                if self._we_raised:
                    if 'FOLD' in opp_last:
                        self.opponent.update_fold_to_our_raise(street)
                    elif 'CALL' in opp_last:
                        self.opponent.update_call_to_our_raise(street)
                    self._we_raised = False

                # Track opponent preflop actions for CFR table traversal
                if street == 0:
                    if 'RAISE' in opp_last:
                        ip_map, oop_map, facing_map = _get_preflop_node_map()
                        facing = facing_map.get(self._preflop_history, 2.0)
                        abs_raise = _classify_preflop_raise(float(opp_bet), facing)
                        self._preflop_history = self._preflop_history + (abs_raise,)
                    elif ('CALL' in opp_last) and not self._preflop_history:
                        # Opponent is SB and limped (root CALL) — track for
                        # correct BFS node traversal
                        self._preflop_history = self._preflop_history + (CALL,)

                # Record in-hand action for Bayesian range narrowing (post-flop only)
                if street >= 1:
                    action_key = ('raise' if 'RAISE' in opp_last else
                                  'call'  if 'CALL'  in opp_last else
                                  'check' if 'CHECK' in opp_last else
                                  'fold'  if 'FOLD'  in opp_last else None)
                    if action_key:
                        self._hand_actions.append((action_key, street))

                # Advance solver on opponent actions (all post-flop streets)
                if street >= 1 and self._solver._initialized:
                    try:
                        pot_for_abs = my_bet + opp_bet
                        abs_act = abstract_action_from_real(
                            opp_last.split('_')[0] if '_' in opp_last else opp_last,
                            opp_raise_size, pot_for_abs, max_raise or 200
                        )
                        self._solver.observe_action(abs_act)
                    except Exception as e:
                        self.logger.info(f"H{self._hands_played} SOLVER-OBS-ERR {e}")

        # Update board and cards
        self._my_cards = my_cards
        self._board = community

        # Notify solver of new board cards (turn/river)
        # Skip solver work when both all-in (no decisions left, save time budget)
        real_pot = my_bet + opp_bet
        real_stack = max_raise + max(my_bet, opp_bet) if max_raise and max_raise > 0 else 0
        is_all_in = (real_stack <= 0)

        if street == 2 and self._last_street < 2 and len(community) >= 4:
            if self._solver._initialized and not is_all_in:
                try:
                    _t_solve = time.perf_counter()
                    self._solver.update_for_new_street(
                        community[3], real_pot=real_pot, real_stack=real_stack,
                        hand_actions=self._hand_actions if self._hand_actions else None,
                        opp_profile=self.opponent)
                    self.logger.info(f"H{self._hands_played} s2 SOLVE-TIME ms={(time.perf_counter()-_t_solve)*1000:.0f}")
                except Exception as e:
                    self.logger.info(f"H{self._hands_played} SOLVER-TURN-ERR {e}")
            elif is_all_in:
                self.logger.info(f"H{self._hands_played} s2 SKIP-SOLVER (all-in)")

        if street == 3 and self._last_street < 3 and len(community) >= 5:
            if self._solver._initialized and not is_all_in:
                try:
                    _t_solve = time.perf_counter()
                    self._solver.update_for_new_street(
                        community[4], real_pot=real_pot, real_stack=real_stack,
                        hand_actions=self._hand_actions if self._hand_actions else None,
                        opp_profile=self.opponent)
                    self.logger.info(f"H{self._hands_played} s3 SOLVE-TIME ms={(time.perf_counter()-_t_solve)*1000:.0f}")
                except Exception as e:
                    self.logger.info(f"H{self._hands_played} SOLVER-RIVER-ERR {e}")
            elif is_all_in:
                self.logger.info(f"H{self._hands_played} s3 SKIP-SOLVER (all-in)")

        self._street = street
        self._last_street = street

        # Track opponent discards
        if len(opp_discards) == 3 and self._opp_discards is None:
            self._opp_discards = tuple(opp_discards)
            self.opponent.update_discards(opp_discards)
            # Update solver's opponent range filter with observed discards
            if self._solver._initialized:
                try:
                    self._solver.update_opponent_discards(tuple(opp_discards))
                except Exception as e:
                    self.logger.info(f"H{self._hands_played} SOLVER-DISCARD-ERR {e}")

        # Track our discards
        if len(my_discards_obs) == 3 and self._my_discards is None:
            self._my_discards = tuple(my_discards_obs)

        # Lockout check
        match_state = self.match_mgr.get_state()
        if match_state['in_lockout']:
            self.logger.info(f"H{self._hands_played} LOCKOUT lead={match_state['cumulative_reward']:+.0f}")
            if valid[2]:
                return (2, 0, 0, 0)
            return (0, 0, 0, 0)

        # DISCARD action (street 1, first action)
        if valid[4]:
            hole_5 = tuple(my_cards[:5])
            flop_3 = tuple(community[:3])
            weights_fn = self.opponent.opp_weights_fn
            keep_i, keep_j = optimal_discard(hole_5, flop_3, weights_fn)
            self._my_kept = (hole_5[keep_i], hole_5[keep_j])
            self.logger.info(
                f"H{self._hands_played} s{street} DISCARD "
                f"hole={self._card_str(hole_5)} flop={self._card_str(flop_3)} "
                f"keep=({keep_i},{keep_j})={self._card_str(self._my_kept)} "
                f"cum={match_state['cumulative_reward']:+.0f}"
            )

            # Initialize solver at discard time (no value networks — HS2 proxy)
            try:
                ms = self.match_mgr.get_state()
                margin = ms['cumulative_reward']
                hands_left = self.match_mgr.hands_remaining
                time_left = ms.get('time_left', 1500)

                if time_left >= 30:
                    # V12: Spend more compute. Tournament data shows 15-17%
                    # budget usage (v5 match_86278: 223s / 1499s). ~750 of 1000
                    # hands are prefolded, leaving ~250 real. With 1440s usable
                    # that's ~5.7s per real hand — our old 5.0s cap wasted the
                    # surplus. Raise caps and multipliers so real hands get
                    # enough budget for flop initial solve + observe_action
                    # rebuilds. The fair_share mechanism auto-scales down if
                    # time runs low (e.g., 82% showdown matches).
                    usable = max(0, time_left - 60)
                    fair_share = usable / max(1, hands_left)
                    if hands_left > 700:       # early game
                        budget = min(8.0, fair_share * 6.0)
                    elif hands_left > 400:     # mid game
                        budget = min(6.0, fair_share * 5.0)
                    else:                      # late game
                        budget = min(4.0, fair_share * 3.0)
                    # Compute preflop range narrowing from opponent's preflop actions
                    preflop_weights = self._compute_preflop_likelihoods(in_position)
                    if preflop_weights is not None:
                        nz = np.count_nonzero(preflop_weights > 0.02)
                        self._range_collapsed = (nz == 0)
                        self.logger.info(
                            f"H{self._hands_played} PREFLOP-NARROW "
                            f"hist={self._preflop_history} "
                            f"active_hands={nz}/80730 "
                            f"max_w={preflop_weights.max():.4f}")
                    else:
                        self._range_collapsed = False

                    _t_solve = time.perf_counter()
                    self._solver.initialize(
                        our_5_cards=tuple(hole_5),
                        flop_3=tuple(community[:3]),
                        pot=my_bet + opp_bet,
                        stack=max_raise + max(my_bet, opp_bet) if max_raise > 0 else 100,
                        opp_profile=self.opponent,
                        match_margin=margin,
                        hands_remaining=hands_left,
                        budget_seconds=budget,
                        value_net_flop=self._value_net_flop,
                        value_net_turn=self._value_net_turn,
                        value_net_river=self._value_net_river,
                        in_position=in_position,
                        opp_discards=self._opp_discards,
                        hand_actions=self._hand_actions if self._hand_actions else None,
                        preflop_weights=preflop_weights,
                        our_kept=self._my_kept,
                    )
                    self.logger.info(f"H{self._hands_played} s1 SOLVE-TIME ms={(time.perf_counter()-_t_solve)*1000:.0f}")
            except Exception as e:
                self.logger.info(f"H{self._hands_played} SOLVER-INIT-ERR {e}")

            return (4, 0, keep_i, keep_j)

        # BETTING action — fast-path for all-in runouts (no decisions left)
        if max_raise == 0 and opp_bet <= my_bet:
            self.logger.info(
                f"H{self._hands_played} s{street} CHECK (all-in runout)")
            return (2, 0, 0, 0)  # CHECK — only legal action

        if self._my_kept is not None:
            my_2 = self._my_kept
        elif len(my_cards) == 2:
            my_2 = tuple(my_cards)
        else:
            my_2 = tuple(my_cards[:2])

        pot = my_bet + opp_bet
        cost = opp_bet - my_bet
        pot_odds = cost / max(1, pot + cost)
        facing_raise = cost > 0

        # Compute equity
        dead_cards = list(self._my_discards or []) + list(self._opp_discards or [])
        if street == 0:
            eq = preflop_strength(tuple(my_cards[:5])) if len(my_cards) == 5 else 0.5
        else:
            eq = exact_equity(
                my_2, tuple(community), tuple(dead_cards),
                opp_discards=tuple(self._opp_discards) if self._opp_discards and all(c != -1 for c in self._opp_discards) else None,
                my_discards=tuple(self._my_discards) if self._my_discards else None,
                opp_weights_fn=self.opponent.opp_weights_fn,
                hand_actions=self._hand_actions if self._hand_actions else None,
            )

        # Get opponent context
        opp_ctx = self.opponent.get_context(street)

        # V10.1: Equity fold gate REMOVED. Let CFR decide all postflop actions.
        # The gate was overriding solver decisions and destroying implied odds.
        # DeepStack/Libratus never override the solver with raw equity.

        # Strategy decision
        action = self._decide_with_solver_or_fallback(
            valid, min_raise, max_raise, pot, eq, pot_odds, cost,
            street, opp_ctx, match_state, community, facing_raise, in_position
        )

        # SAFETY: Never fold when checking is free (cost == 0)
        if action[0] == 0 and cost <= 0 and valid[2]:
            self.logger.info(
                f"H{self._hands_played} s{street} RAIL name=no_fold_free "
                f"orig=FOLD final=CHECK eq={eq:.3f}"
            )
            action = (2, 0, 0, 0)

        # V9.2: Removed no_fold_profitable rail. It was overriding 3700+ CFR folds
        # per tournament, defeating the implied-odds correction. Trust the solver.

        # Log the decision
        act_name = ACTION_NAMES.get(action[0], f'?{action[0]}')
        extra = f" amt={action[1]}" if action[0] == 1 else ""
        self.logger.info(
            f"H{self._hands_played} s{street} {act_name}{extra} "
            f"eq={eq:.3f} pot={pot} cost={cost} odds={pot_odds:.2f} "
            f"agg={match_state['aggression']:.2f} prs={match_state['pressure']:+.2f} "
            f"fr={opp_ctx.get('fold_rate_to_our_raise', 0):.2f} "
            f"br={opp_ctx.get('bluff_rate', 0):.2f} "
            f"opp={opp_ctx.get('opponent_type', '?')} "
            f"ip={int(in_position)} "
            f"hand={self._card_str(my_2)} board={self._card_str(community)} "
            f"cum={match_state['cumulative_reward']:+.0f}"
        )

        # Track if we raised
        if action[0] == 1:
            self._we_raised = True

        return action

    def _decide_with_solver_or_fallback(self, valid, min_raise, max_raise, pot, eq,
                                         pot_odds, cost, street, opp_ctx, match_state,
                                         community, facing_raise, in_position):
        """V9.3: CFR for fold/call + equity-based raise exploitation.

        Root cause of V9.1-V9.2 calling station: CFR computes Nash equilibrium,
        which slow-plays strong hands (opponent folds to raises in equilibrium).
        But real opponents are heuristic: they call too much and rarely bluff.
        Fix: trust CFR fold decisions (implied-odds correct), but override
        call→raise with equity-based exploitation.
        """
        # Preflop: solved CFR table lookup, heuristic fallback
        if street == 0:
            cfr_action = self._decide_preflop_cfr(
                tuple(self._my_cards[:5]) if len(self._my_cards) >= 5 else tuple(self._my_cards),
                in_position, valid, min_raise, max_raise, pot, cost, match_state,
            )
            if cfr_action is not None:
                return cfr_action
            # Fallback to heuristic when no CFR table / no matching node
            heuristic_action = decide(
                equity=eq, pot_odds=pot_odds, street=street, pot=pot,
                cost=cost, min_raise=min_raise, max_raise=max_raise,
                valid_actions=valid, opp_ctx=opp_ctx, match_state=match_state,
                board=community, persona={},
                facing_raise=facing_raise, in_position=in_position,
            )
            return heuristic_action

        try:
            solver_ready = (self._solver._initialized and
                            self._solver._iterations_run >= 50)

            if not solver_ready:
                # Minimal fallback: no heuristic, just equity-based
                return self._equity_fallback(valid, eq, pot_odds, cost, facing_raise, min_raise, max_raise, pot)

            cfvs = self._solver.get_action_cfvs()
            avg_strat = self._solver.get_root_average_strategy()
            iters = getattr(self._solver, '_iterations_run', 0)
            self.logger.info(
                f"H{self._hands_played} s{street} CFR-CFVS "
                f"fold={cfvs[0]:.4f} call={cfvs[1]:.4f} "
                f"rs={cfvs[2]:.4f} rl={cfvs[3]:.4f} ra={cfvs[4]:.4f} "
                f"iters={iters}"
            )
            mode = self.match_mgr.get_strategy_mode()

            # Fix #24: LAG dampening removed. Protection/pressure modes are now
            # probabilistic (biased sampling) instead of deterministic, so
            # pressure mode no longer always jams ALL-IN vs LAGs.

            # Build valid abstract actions
            abstract_valid = []
            if valid[0]: abstract_valid.append(FOLD)
            if valid[2] or valid[3]: abstract_valid.append(CALL)
            if valid[1] and max_raise >= min_raise:
                abstract_valid.extend([RAISE_SMALL, RAISE_LARGE, RAISE_OVERBET, RAISE_ALLIN])
            abstract_valid = list(dict.fromkeys(abstract_valid))

            if not abstract_valid:
                return self._equity_fallback(valid, eq, pot_odds, cost, facing_raise, min_raise, max_raise, pot)

            # Redistribute raise mass to CALL when raise is invalid
            can_raise = any(a in ALL_RAISES for a in abstract_valid)
            if not can_raise and CALL in abstract_valid:
                adj = avg_strat.copy()
                raise_mass = sum(float(adj[a]) for a in ALL_RAISES)
                if raise_mass > 0:
                    adj[CALL] += raise_mass
                    for a in ALL_RAISES:
                        adj[a] = 0
                    avg_strat = adj

            # Action selection: always pass avg_strategy to select_action.
            # When CFVs clearly distinguish actions (spread > epsilon),
            # select_action uses CFV argmax. When CFVs are close or zero,
            # it falls back to avg_strategy (mixed Nash equilibrium).
            # Fix #16: pure CFV argmax without avg_strategy fallback caused
            # 4% fold rate (vs 22% with avg_strategy), because CFVs
            # systematically overestimate CALL when opponent range isn't
            # properly narrowed.
            abstract_act = select_action(cfvs, abstract_valid, mode,
                                         self._solver._iterations_run,
                                         avg_strategy=avg_strat)

            # Fix #30: Range collapse guard.
            # When preflop narrowing yields 0 active opponent hands, CFVs
            # are computed against a flat/default range — unreliable.
            # Fall back to equity-based play: only raise with strong equity.
            if self._range_collapsed and abstract_act in ALL_RAISES:
                if eq < 0.55:
                    abstract_act = CALL if CALL in abstract_valid else abstract_act
                    self.logger.info(
                        f"H{self._hands_played} s{street} RANGE-COLLAPSE-GUARD "
                        f"eq={eq:.3f} downgrade to CALL")

            # Fix #29: Postflop all-in guard.
            # v10 data: 46% of flop raises are all-in. 24 shoves at <30% eq
            # lost 455 chips. Suppress all-in when equity is low and CFV
            # doesn't strongly support it.
            if abstract_act == RAISE_ALLIN and eq < 0.35:
                # Only allow if CFV for all-in is the best or within 0.002
                best_cfv = max(float(cfvs[a]) for a in abstract_valid)
                allin_cfv = float(cfvs[RAISE_ALLIN])
                if allin_cfv < best_cfv - 0.002:
                    # Downgrade to RAISE_LARGE if available, else CALL
                    if RAISE_LARGE in abstract_valid:
                        abstract_act = RAISE_LARGE
                    elif CALL in abstract_valid:
                        abstract_act = CALL
                    self.logger.info(
                        f"H{self._hands_played} s{street} ALLIN-GUARD "
                        f"eq={eq:.3f} allin_cfv={allin_cfv:.4f} best={best_cfv:.4f}")

            # Fix #18: Tightened equity sanity check.
            # V8 logs show equity overestimation persists — our 30-41% actual
            # SD win rate vs 50-70% estimated. Removing the 0.05 cushion and
            # adding a 0.03 margin so we need a genuine edge to continue.
            if (facing_raise and cost > 0
                    and abstract_act in (CALL,) + ALL_RAISES
                    and eq < pot_odds + 0.03
                    and FOLD in abstract_valid):
                self.logger.info(
                    f"H{self._hands_played} s{street} SANITY "
                    f"fold eq={eq:.3f} < pot_odds+0.03={pot_odds + 0.03:.2f}")
                abstract_act = FOLD

            # Fold guard removed: pass-2 now corrects ALL terminal values
            # (depth-limit + river showdown) to be hand-specific using equity.
            # The solver's pot-odds calculation is correct with accurate terminals.

            action = self._abstract_to_game_action(
                abstract_act, valid, min_raise, max_raise, pot, cost)

            # Advance solver tree
            solver_act = abstract_act
            if abstract_act == FOLD and cost <= 0:
                solver_act = CALL
            try:
                self._solver.observe_action(solver_act)
            except Exception:
                pass

            # Diagnostics
            rhs = self._solver.get_our_rhs()
            opp_dist = self._solver.get_opp_class_distribution()
            sorted_cfvs = sorted(cfvs[abstract_valid], reverse=True)
            util_gap = (sorted_cfvs[0] - sorted_cfvs[1]) if len(sorted_cfvs) > 1 else 0.0

            self.logger.info(
                f"H{self._hands_played} s{street} CFR act={abstract_act} "
                f"iters={self._solver._iterations_run} eq={eq:.3f} rhs={rhs:.3f} "
                f"gap={util_gap:.4f} dist={opp_dist}")
            return action

        except Exception as e:
            self.logger.info(f"H{self._hands_played} s{street} CFR-ERR {e}")

        return self._equity_fallback(valid, eq, pot_odds, cost, facing_raise, min_raise, max_raise, pot)

    def _decide_preflop_cfr(self, hand_cards, in_position, valid, min_raise,
                            max_raise, pot, cost, match_state):
        """Look up preflop strategy from solved CFR tables.

        Returns game action tuple or None if no matching table/node.
        """
        if not self._preflop_tables or len(hand_cards) != 5:
            return None

        # Always use neutral preflop tables. Protection tables produce 3.7x
        # more all-in mass (31.5% vs 8.5% OOP), creating variance that erodes
        # chip leads. Pressure tables have the same issue (69% all-in avg).
        # Superhuman AIs use linear chip-EV; tournament utility distortion
        # belongs in postflop only.
        variant = 'neutral'
        if variant not in self._preflop_tables:
            variant = next(iter(self._preflop_tables), None)
        if variant is None:
            return None

        hand_idx = combo_index_5(tuple(sorted(hand_cards)))

        ip_map, oop_map, _ = _get_preflop_node_map()
        node_map = ip_map if in_position else oop_map

        if self._preflop_history not in node_map:
            return None  # no matching node (e.g., opponent limped)

        node_idx = node_map[self._preflop_history]
        table_key = 'ip' if in_position else 'oop'
        table = self._preflop_tables[variant][table_key]

        if hand_idx >= table.shape[0] or node_idx >= table.shape[1]:
            return None

        strategy = table[hand_idx, node_idx].copy()  # (5,) probabilities

        # Mask unavailable actions and renormalize
        mask = np.zeros(5, dtype=np.float32)
        if valid[0]:
            mask[FOLD] = 1.0
        if valid[3] or valid[2]:
            mask[CALL] = 1.0
        if valid[1] and max_raise >= min_raise:
            mask[RAISE_SMALL] = 1.0
            mask[RAISE_LARGE] = 1.0
            mask[RAISE_ALLIN] = 1.0

        strategy *= mask
        total = strategy.sum()
        if total < 1e-8:
            return None
        strategy /= total

        # --- Gap 2 fix: EV-based opponent-adaptive preflop reweighting ---
        # Static tables assume balanced opponent. Real opponents deviate.
        # Compute EV of terminal actions (FOLD, ALL-IN) exactly, use
        # opponent fold rate to adjust non-terminal raise frequency.
        eq = preflop_strength(hand_cards) if len(hand_cards) == 5 else 0.5
        opp_ctx = self.opponent.get_context(0)
        pf_fold_rate = opp_ctx.get('preflop_fold_rate', 0.3)
        total_actions = opp_ctx.get('total_actions', 0)
        shove_ev = pf_fold_rate * pot + (1 - pf_fold_rate) * (200 * eq - 100)

        if total_actions >= 10:
            # --- ALL-IN: exact EV ---
            # EV = f*pot + (1-f)*(200*eq - 100)
            shove_ev = pf_fold_rate * pot + (1 - pf_fold_rate) * (200 * eq - 100)

            if strategy[RAISE_ALLIN] > 0.01:
                if shove_ev < 0:
                    # Negative EV shove: dampen exponentially
                    # At -5 EV → 0.37x, at -10 EV → 0.14x, at -20 EV → 0.02x
                    dampen = float(np.exp(shove_ev / 5.0))
                    strategy[RAISE_ALLIN] *= dampen
                elif shove_ev > 5:
                    # Very +EV shove (high fold equity): boost up to 2x
                    boost = min(2.0, 1.0 + shove_ev / 10.0)
                    strategy[RAISE_ALLIN] *= boost

            # --- CALL vs big raise: pot odds check ---
            eff_stack = max_raise + cost if max_raise > 0 else 100
            if cost > 0 and strategy[CALL] > 0.01:
                pot_odds = cost / (pot + cost)
                if eq < pot_odds - 0.03:
                    # Clear -EV call. Dampen proportionally.
                    gap = pot_odds - eq
                    dampen = float(np.exp(-gap * 15.0))  # gap=0.10 → 0.22x
                    strategy[CALL] *= dampen

            # --- Raise frequency vs opponent fold rate ---
            if pf_fold_rate > 0.50:
                # High fold equity: boost raises, suppress folds
                raise_bonus = 1.0 + (pf_fold_rate - 0.50) * 2.0
                strategy[RAISE_SMALL] *= raise_bonus
                strategy[RAISE_LARGE] *= raise_bonus
                strategy[FOLD] *= max(0.1, 1.0 - pf_fold_rate)
            elif pf_fold_rate < 0.20:
                # Station: fold equity is near zero. Bluff raises are -EV.
                # Suppress raises with weak hands, boost calls.
                if eq < 0.52:
                    strategy[RAISE_SMALL] *= 0.7
                    strategy[RAISE_LARGE] *= 0.7
                strategy[CALL] *= 1.3

            # Renormalize
            total = strategy.sum()
            if total < 1e-8:
                return None
            strategy /= total

        abstract_act = int(np.random.choice(5, p=strategy))

        # Track our preflop action for correct BFS node traversal
        if abstract_act in ALL_RAISES:
            self._preflop_history = self._preflop_history + (abstract_act,)
        elif abstract_act == CALL and not self._preflop_history:
            # SB limp (root CALL) — creates non-terminal node in BFS tree
            self._preflop_history = self._preflop_history + (CALL,)

        self.logger.info(
            f"H{self._hands_played} s0 PREFLOP-CFR "
            f"var={variant} node={node_idx} act={abstract_act} "
            f"eq={eq:.3f} pf_fold={pf_fold_rate:.2f} cost={cost} "
            f"shove_ev={shove_ev:+.1f} "
            f"strat=[{strategy[0]:.2f},{strategy[1]:.2f},{strategy[2]:.2f},"
            f"{strategy[3]:.2f},{strategy[4]:.2f}] "
            f"hist={self._preflop_history}"
        )

        return self._abstract_to_game_action(
            abstract_act, valid, min_raise, max_raise, pot, cost)

    def _equity_fallback(self, valid, eq, pot_odds, cost, facing_raise,
                         min_raise=0, max_raise=0, pot=0):
        """Equity-based fallback when CFR unavailable."""
        if cost <= 0:
            # Not facing a raise: bet for value with strong hands
            if eq > 0.70 and valid[1] and max_raise >= min_raise > 0:
                # Value bet ~60% pot, clamped to [min_raise, max_raise]
                bet = max(min_raise, min(max_raise, int(pot * 0.6)))
                return (1, bet, 0, 0)
            return (2, 0, 0, 0) if valid[2] else (3, 0, 0, 0)
        # Facing a raise
        if eq > 0.75 and valid[1] and max_raise >= min_raise > 0:
            # Strong hand facing raise — re-raise
            reraise = max(min_raise, min(max_raise, int(pot * 0.75)))
            return (1, reraise, 0, 0)
        if eq > pot_odds + 0.05:
            return (3, 0, 0, 0) if valid[3] else (0, 0, 0, 0)
        return (0, 0, 0, 0) if valid[0] else (3, 0, 0, 0)

    def _abstract_to_game_action(self, abstract_act, valid, min_raise, max_raise, pot, cost=0):
        """Map abstract CFR action to concrete game action tuple."""
        if abstract_act == FOLD and valid[0]:
            return (0, 0, 0, 0)
        if abstract_act == CALL:
            if valid[2]:
                return (2, 0, 0, 0)
            if valid[3]:
                return (3, 0, 0, 0)
        if abstract_act in ALL_RAISES and valid[1]:
            current_bet = cost if not valid[2] else 0
            amt = resolve_raise_amount(abstract_act, pot, max_raise,
                                       current_bet=current_bet)
            amt = max(min_raise, min(max_raise, int(amt)))
            return (1, amt, 0, 0)
        # Safe fallback
        if valid[2]: return (2, 0, 0, 0)
        if valid[3]: return (3, 0, 0, 0)
        return (0, 0, 0, 0)

    def observe(self, observation, reward, terminated, truncated, info):
        """Called after opponent's action when it's not our turn, and at hand end."""
        if terminated:
            # Detect opponent fold to our raise
            if self._we_raised and 'player_0_cards' not in info and reward > 0:
                self.opponent.update_fold_to_our_raise(max(0, self._street))

            self.match_mgr.update(
                reward=reward,
                time_used=observation.get('time_used', 0),
                time_left=observation.get('time_left', 0),
            )

            # Process showdown if available
            showdown_info = ""
            if 'player_0_cards' in info:
                self._process_showdown(info, observation)
                if self._we_raised:
                    self.opponent.update_call_to_our_raise(max(0, self._street))
                showdown_info = " SHOWDOWN"

            # Log hand result
            ms = self.match_mgr.get_state()
            tl = observation.get('time_left', 0)
            self.logger.info(
                f"H{self._hands_played} END r={reward:+.0f} "
                f"cum={ms['cumulative_reward']:+.0f} "
                f"tl={tl:.0f}{showdown_info}"
            )

            # V7 Laboratory: Log to SQLite
            try:
                opp_ctx = self.opponent.get_context(max(0, self._street))
                self.sql_logger.log_hand(
                    match_id=ms.get('match_id', 'local'),
                    hand_num=self._hands_played,
                    bot_version="v10",
                    pnl=reward,
                    cum=ms['cumulative_reward'],
                    our_c=self._my_kept if self._my_kept else self._my_cards,
                    board=self._board,
                    opp_c=self._opp_showdown_cards,
                    opp_t=opp_ctx.get('opponent_type', '?'),
                    street=self._street,
                    is_sd=(showdown_info != ""),
                    lockout=ms['in_lockout']
                )
            except Exception:
                pass

            self.opponent.reset_hand()
            self._reset_hand_state()
            self._hands_played += 1

    def _process_showdown(self, info, obs):
        """Extract opponent hand info from showdown and update model."""
        from gym_env import PokerEnv
        try:
            community = [c for c in obs['community_cards'] if c != -1]
            if len(community) != 5:
                return

            my_cards = [c for c in obs['my_cards'] if c != -1]

            p0_cards = info.get('player_0_cards', [])
            p1_cards = info.get('player_1_cards', [])

            my_strs = set(PokerEnv.int_card_to_str(c) for c in my_cards)

            if set(p0_cards) == my_strs:
                opp_strs = p1_cards
            elif set(p1_cards) == my_strs:
                opp_strs = p0_cards
            else:
                return

            if len(opp_strs) != 2:
                return

            self._opp_showdown_cards = opp_strs
            from treys import Card
            opp_treys = [Card.new(s) for s in opp_strs]
            board_treys = [PokerEnv.int_to_card(c) for c in community]
            opp_score = self._evaluator.evaluate(opp_treys, board_treys)

            hand_type = classify_hand(opp_score)
            is_strong = hand_type in ('straight_flush', 'full_house', 'flush', 'straight', 'trips')

            self.logger.info(
                f"H{self._hands_played} SHOWDOWN-OPP {opp_strs} "
                f"type={hand_type} score={opp_score} strong={is_strong}"
            )

            self.opponent.update_showdown(
                raised=self.opponent._raised_this_hand,
                hand_strong=is_strong,
                hand_score=opp_score,
            )
        except Exception:
            pass

    def _reset_hand_state(self):
        self._my_cards = []
        self._my_kept = None
        self._my_discards = None
        self._board = []
        self._opp_discards = None
        self._opp_showdown_cards = None
        self._street = -1
        self._last_street = -1
        self._we_raised = False
        self._hand_actions = []
        self._preflop_history = ()
        self._solver._initialized = False


if __name__ == "__main__":
    PlayerAgent.run()
