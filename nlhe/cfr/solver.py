"""
NLHESolver -- CFR-D Subgame Solver for 52-card No-Limit Hold'em.

Full game-tree CFR with:
- Recursive tree construction for one street of betting
- Per-hand (1326,) strategies at every decision node
- DCFR discounting (Brown & Sandholm 2019)
- Phantom range widening for hero
- RNR blend for opponent model incorporation
- Pass-2 best-response extraction for hand-specific CFVs
- Variance-controlled action selection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nlhe.cfr.abstraction import (
    ALL_HANDS,
    HAND_TO_IDX,
    HAND_CONTAINS_CARD,
    N_ACTIONS,
    ACTION_NAMES,
    FOLD,
    CALL,
    RAISE_SMALL,
    RAISE_LARGE,
    RAISE_ALLIN,
    RAISE_OVERBET,
    resolve_raise_amount,
)
from nlhe.cfr.equity import (
    card_str_to_idx,
    equity_vs_range_batch,
)
from nlhe.game import (
    GameState,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_HANDS = 1326
_RAISE_CAP = 2  # max raises per street
_DCFR_ALPHA = 1.5
_DCFR_BETA = 0.0
_DCFR_GAMMA = 2.0

# Terminal type constants
_TERM_FOLD_P1 = 0   # P1 folded
_TERM_FOLD_P2 = 1   # P2 folded
_TERM_SHOWDOWN = 2  # showdown
_TERM_DEPTH = 3     # depth-limit boundary


# ---------------------------------------------------------------------------
# SubgameNode
# ---------------------------------------------------------------------------

@dataclass
class SubgameNode:
    """A node in the subgame tree for one street of betting."""
    player: int           # 0 = P1 (hero), 1 = P2 (opponent), -1 = terminal
    pot: float
    stack: float          # effective remaining stack
    ip_in_pot: float      # hero's contribution this street
    oop_in_pot: float     # opponent's contribution this street
    action_history: tuple
    streets_remaining: int
    children: dict = field(default_factory=dict)

    # Per-hand arrays for decision nodes
    regret: Optional[np.ndarray] = None
    strategy_sum: Optional[np.ndarray] = None

    # Terminal info
    terminal_type: int = -1       # one of _TERM_* constants
    terminal_hero_pot: float = 0  # hero's chips at risk (for fold terminals)
    terminal_opp_pot: float = 0   # opp's chips at risk (for fold terminals)

    _flat_idx: int = -1
    _child_keys: Optional[list] = None


# ---------------------------------------------------------------------------
# NLHESolver
# ---------------------------------------------------------------------------

class NLHESolver:
    """
    Full game-tree CFR solver for a single NLHE decision point.

    Instantiate once per decision. Call solve() to get the action string.
    Call observe_action() after the opponent acts to update their range.
    """

    def __init__(
        self,
        state: GameState,
        our_hole: list[str],
        budget_seconds: float = 5.0,
        opponent_model=None,
        initial_opp_range: np.ndarray | None = None,
    ) -> None:
        self.state = state
        self.our_hole = our_hole
        self.budget = budget_seconds
        self._iterations = 0

        # Convert hole cards to index
        self.our_hole_idx: tuple[int, int] = tuple(
            sorted(card_str_to_idx(c) for c in our_hole)
        )

        # Dead cards: our hole + board (for opponent range)
        self._dead_idxs = self._compute_dead_card_idxs(state)
        self._dead_mask = np.zeros(_N_HANDS, dtype=bool)
        for c_idx in self._dead_idxs:
            self._dead_mask |= HAND_CONTAINS_CARD[c_idx]
        self._live_mask = ~self._dead_mask

        # Board-only dead mask (for hero equity/CFV — hero CAN hold their own cards)
        self._board_dead_mask = np.zeros(_N_HANDS, dtype=bool)
        for c in state.board:
            self._board_dead_mask |= HAND_CONTAINS_CARD[card_str_to_idx(c)]

        # Opponent model
        if opponent_model is None:
            from nlhe.cfr.opponent import OpponentModel
            self._opp_model = OpponentModel()
        else:
            self._opp_model = opponent_model

        # Opponent range: use carried-over range if available (cross-street
        # Bayesian tracking), otherwise build from scratch.
        if initial_opp_range is not None:
            self.opp_range = initial_opp_range.copy()
            # Re-mask for current dead cards (new board cards since last street)
            for c_idx in self._dead_idxs:
                self.opp_range[HAND_CONTAINS_CARD[c_idx]] = 0.0
            total = self.opp_range.sum()
            if total > 0:
                self.opp_range /= total
            else:
                self.opp_range = self._build_initial_range(state)
        else:
            self.opp_range = self._build_initial_range(state)

        # Hero range: phantom-widened point mass
        self._hero_range = self._build_phantom_range()

        # Cache action list
        self._valid_actions: list[str] = list(state.valid_actions)

        # Our hand index in the 1326-hand array
        self._our_hand_idx: int = HAND_TO_IDX.get(self.our_hole_idx, 0)

        # Internal state
        self._avg_strategy: Optional[np.ndarray] = None
        self._root: Optional[SubgameNode] = None
        self._flat_nodes: list[SubgameNode] = []
        self._equity_cache: Optional[np.ndarray] = None  # lazy (1326,) equity
        self._hero_matchup: Optional[np.ndarray] = None  # lazy (1326,) hero vs each opp hand
        self._last_action: Optional[str] = None  # hero's last chosen action

        # Build tree if multiple actions
        if len(self._valid_actions) > 1:
            self._build_tree_and_prepare()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> str:
        """Run DCFR iterations until budget expires, then select an action."""
        actions = self._valid_actions
        n_actions = len(actions)

        if n_actions == 0:
            raise ValueError("No valid actions in state")

        if n_actions == 1:
            self._avg_strategy = np.array([1.0], dtype=np.float32)
            return actions[0]

        # Ensure equity is computed (lazy)
        self._ensure_equity()

        # Run DCFR iterations
        deadline = time.monotonic() + self.budget
        t = 0
        while time.monotonic() < deadline:
            self._cfr_iteration(t)
            t += 1
        self._iterations = t

        # Extract per-action CFVs for our hand (pass 2)
        action_cfvs = self._extract_hand_cfvs()

        # Get average strategy at root for our hand
        avg_strategy = self._get_root_avg_strategy()
        self._avg_strategy = avg_strategy

        # Pure equilibrium sampling (no heuristic variance control bandage)
        probs = avg_strategy.astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(actions), dtype=np.float64) / len(actions)
            
        action_idx = int(np.random.choice(len(actions), p=probs))

        chosen = self._valid_actions[action_idx]
        self._last_action = chosen

        # Diagnostic info (accessible externally)
        self._diag = {
            'iterations': t,
            'avg_strategy': {a: f'{s:.3f}' for a, s in zip(actions, avg_strategy)},
            'cfvs': {a: f'{v:.1f}' for a, v in zip(actions, action_cfvs)},
            'chosen': chosen,
        }

        return chosen

    def observe_action(self, action: str) -> None:
        """Update opp_range after the opponent acts using Bayesian range narrowing.

        Uses the OpponentModel's robust hand-strength based likelihoods.
        This prevents the range from being corrupted by 1-street CFR depth-limit
        artifacts when tracking ranges across multiple streets.
        """
        for c_idx in self._dead_idxs:
            self.opp_range[HAND_CONTAINS_CARD[c_idx]] = 0.0

        strength = self._estimate_opp_hand_strength(self.state)
        
        # Map raw action to a generalized action for the model
        mapped_action = action
        if action.startswith('RAISE'):
            # Estimate bet size roughly if not explicitly ALLIN or OVERBET
            if action not in ('RAISE_ALLIN', 'RAISE_OVERBET'):
                mapped_action = 'RAISE_LARGE' # default safe assumption
        
        likelihoods = self._opp_model.get_range_weights(strength, mapped_action)
        self.opp_range *= likelihoods
        
        total = self.opp_range.sum()
        if total > 0.0:
            self.opp_range /= total
        else:
            # Fallback if range becomes empty
            self.opp_range = self._build_initial_range(self.state)

    def get_opp_action_likelihoods(
        self, hero_action_str: str | None, opp_action_str: str
    ) -> np.ndarray:
        """Get P(opp_action | hand) from the solver's equilibrium strategies.

        Navigates the tree: root -> hero's action child -> P2 node.
        Returns (1326,) array of per-hand likelihoods.
        """
        if (self._root is None or self._root.strategy_sum is None
                or hero_action_str is None):
            return np.ones(_N_HANDS, dtype=np.float32)

        hero_act_int = _action_str_to_int(hero_action_str)
        if hero_act_int not in self._root.children:
            return np.ones(_N_HANDS, dtype=np.float32)

        opp_node = self._root.children[hero_act_int]
        if opp_node.player != 1 or opp_node.strategy_sum is None:
            return np.ones(_N_HANDS, dtype=np.float32)

        # Compute average strategy at this P2 node
        totals = opp_node.strategy_sum.sum(axis=1, keepdims=True)
        n_child = opp_node.strategy_sum.shape[1]
        has_data = (totals.ravel() > 0)
        avg_strat = np.full((_N_HANDS, n_child), 1.0 / n_child, dtype=np.float32)
        avg_strat[has_data] = opp_node.strategy_sum[has_data] / totals[has_data]

        # Map opponent's action to the P2 node's action index
        opp_act_int = _action_str_to_int(opp_action_str)
        child_keys = opp_node._child_keys or []

        if opp_act_int not in child_keys:
            # Any raise maps to whatever raise is in the tree
            if opp_action_str.startswith('RAISE'):
                for k in child_keys:
                    if k in (RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, RAISE_OVERBET):
                        opp_act_int = k
                        break
            if opp_act_int not in child_keys:
                return np.ones(_N_HANDS, dtype=np.float32)

        a_idx = child_keys.index(opp_act_int)
        # Floor at 0.05 to avoid zeroing out hands completely
        return np.maximum(avg_strat[:, a_idx], 0.05)

    # ------------------------------------------------------------------
    # Lazy equity computation
    # ------------------------------------------------------------------

    def _ensure_equity(self) -> None:
        """Compute 1326x1326 matchup matrix once and cache it. Used by all terminal nodes."""
        if hasattr(self, '_matchup_matrix') and self._matchup_matrix is not None:
            return

        from nlhe.cfr.equity import compute_matchup_matrix
        board = self.state.board
        # For river use exact evaluation, for others use Monte Carlo
        n_samples = 30 if len(board) >= 3 else 15
        self._matchup_matrix = compute_matchup_matrix(board, n_samples=n_samples)

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree_and_prepare(self) -> None:
        """Build game tree, flatten it, and tag terminal nodes."""
        state = self.state
        pot = float(state.pot)
        stack = float(state.our_stack)
        streets_remaining = max(0, STREET_RIVER - state.street)

        # Outstanding bet: opponent has bet to_call more than hero this street.
        # ip_in_pot = 0 (hero hasn't acted in the tree yet),
        # oop_in_pot = to_call (the gap hero must close to call).
        ip_in_pot = 0.0
        oop_in_pot = float(getattr(state, 'to_call', 0))

        self._root = self._build_node(
            player=0,  # hero acts (it's our decision point)
            pot=pot,
            stack=stack,
            ip_in_pot=ip_in_pot,
            oop_in_pot=oop_in_pot,
            action_history=(),
            streets_remaining=streets_remaining,
            n_raises=0,
            is_root=True,
        )

        self._prepare_flat_traversal()

    def _build_node(
        self,
        player: int,
        pot: float,
        stack: float,
        ip_in_pot: float,
        oop_in_pot: float,
        action_history: tuple,
        streets_remaining: int,
        n_raises: int,
        is_root: bool = False,
    ) -> SubgameNode:
        """Recursively build a betting tree node for one street."""
        node = SubgameNode(
            player=player,
            pot=pot,
            stack=stack,
            ip_in_pot=ip_in_pot,
            oop_in_pot=oop_in_pot,
            action_history=action_history,
            streets_remaining=streets_remaining,
        )

        if is_root:
            # At root, use actual valid_actions from game state
            for act_str in self._valid_actions:
                act_int = _action_str_to_int(act_str)
                child = self._make_child(
                    act_int, player, pot, stack,
                    ip_in_pot, oop_in_pot, action_history,
                    streets_remaining, n_raises,
                )
                if child is not None:
                    node.children[act_int] = child
            return node

        # Non-root: compact action abstraction
        has_bet = (ip_in_pot != oop_in_pot)
        actions = []

        if has_bet:
            actions.append(FOLD)
        actions.append(CALL)

        if n_raises < _RAISE_CAP and stack > 0:
            call_amount = abs(ip_in_pot - oop_in_pot)
            remaining_after_call = stack - call_amount
            
            if remaining_after_call > 0:
                # Always allow ALL-IN
                actions.append(RAISE_ALLIN)
                
                # Check if smaller raises are valid (don't exceed stack)
                pot_after_call = pot + call_amount
                
                # Small raise (~0.33x or 2.2x)
                small_amount = 0.0
                if call_amount > 0:
                    small_amount = call_amount * 2.2
                else:
                    small_amount = pot_after_call * 0.33
                    
                if small_amount < remaining_after_call * 0.85: # Only add if distinct from all-in
                    actions.append(RAISE_SMALL)
                    
                # Large raise (~0.75x or 3x)
                large_amount = 0.0
                if call_amount > 0:
                    large_amount = call_amount * 3.0
                else:
                    large_amount = pot_after_call * 0.75
                    
                if large_amount < remaining_after_call * 0.85 and large_amount > small_amount * 1.5:
                    actions.append(RAISE_LARGE)
            else:
                actions.append(RAISE_ALLIN)

        for act_int in actions:
            child = self._make_child(
                act_int, player, pot, stack,
                ip_in_pot, oop_in_pot, action_history,
                streets_remaining, n_raises,
            )
            if child is not None:
                node.children[act_int] = child

        return node

    def _make_child(
        self,
        action: int,
        player: int,
        pot: float,
        stack: float,
        ip_in_pot: float,
        oop_in_pot: float,
        action_history: tuple,
        streets_remaining: int,
        n_raises: int,
    ) -> Optional[SubgameNode]:
        """Create a child node for the given action."""
        new_history = action_history + (action,)
        has_bet = (ip_in_pot != oop_in_pot)

        if action == FOLD:
            if not has_bet:
                return None
            node = SubgameNode(
                player=-1,
                pot=pot,
                stack=stack,
                ip_in_pot=ip_in_pot,
                oop_in_pot=oop_in_pot,
                action_history=new_history,
                streets_remaining=streets_remaining,
            )
            # Tag who folded
            if player == 0:
                node.terminal_type = _TERM_FOLD_P1
            else:
                node.terminal_type = _TERM_FOLD_P2
            node.terminal_hero_pot = ip_in_pot
            node.terminal_opp_pot = oop_in_pot
            return node

        elif action == CALL:
            if has_bet:
                call_amount = abs(ip_in_pot - oop_in_pot)
                if player == 0:
                    new_ip = max(ip_in_pot, oop_in_pot)
                    new_oop = oop_in_pot
                else:
                    new_ip = ip_in_pot
                    new_oop = max(ip_in_pot, oop_in_pot)
                new_pot = pot + call_amount
                new_stack = stack - call_amount

                if new_stack <= 0:
                    new_stack = 0

                # Call after bet/raise = street ends
                node = SubgameNode(
                    player=-1,
                    pot=new_pot,
                    stack=max(0, new_stack),
                    ip_in_pot=new_ip,
                    oop_in_pot=new_oop,
                    action_history=new_history,
                    streets_remaining=0 if new_stack <= 0 else streets_remaining,
                )
                if new_stack <= 0 or streets_remaining == 0:
                    node.terminal_type = _TERM_SHOWDOWN
                else:
                    node.terminal_type = _TERM_DEPTH
                node.terminal_hero_pot = new_ip
                node.terminal_opp_pot = new_oop
                return node

            else:
                # Check: if both have checked, street ends
                if len(action_history) >= 1 and action_history[-1] == CALL:
                    node = SubgameNode(
                        player=-1,
                        pot=pot,
                        stack=stack,
                        ip_in_pot=ip_in_pot,
                        oop_in_pot=oop_in_pot,
                        action_history=new_history,
                        streets_remaining=streets_remaining,
                    )
                    if streets_remaining == 0:
                        node.terminal_type = _TERM_SHOWDOWN
                    else:
                        node.terminal_type = _TERM_DEPTH
                    node.terminal_hero_pot = ip_in_pot
                    node.terminal_opp_pot = oop_in_pot
                    return node

                # First check: opponent acts next
                return self._build_node(
                    player=1 - player,
                    pot=pot,
                    stack=stack,
                    ip_in_pot=ip_in_pot,
                    oop_in_pot=oop_in_pot,
                    action_history=new_history,
                    streets_remaining=streets_remaining,
                    n_raises=n_raises,
                )

        else:
            # Raise action
            call_amount = abs(ip_in_pot - oop_in_pot) if has_bet else 0
            remaining_after_call = stack - call_amount

            if action == RAISE_ALLIN:
                raise_amount = stack
            else:
                sized = resolve_raise_amount(
                    action, pot + call_amount, max(0, remaining_after_call)
                )
                raise_amount = call_amount + sized
                raise_amount = min(raise_amount, stack)

            if player == 0:
                new_ip = ip_in_pot + raise_amount
                new_oop = oop_in_pot
            else:
                new_ip = ip_in_pot
                new_oop = oop_in_pot + raise_amount

            new_pot = pot + raise_amount
            new_stack = stack - raise_amount
            if new_stack < 0:
                new_stack = 0

            # Opponent responds
            return self._build_node(
                player=1 - player,
                pot=new_pot,
                stack=new_stack,
                ip_in_pot=new_ip,
                oop_in_pot=new_oop,
                action_history=new_history,
                streets_remaining=0 if new_stack <= 0 else streets_remaining,
                n_raises=n_raises + 1,
            )

    # ------------------------------------------------------------------
    # Flat traversal
    # ------------------------------------------------------------------

    def _prepare_flat_traversal(self) -> None:
        """Convert recursive tree to post-order flat array with pre-allocated arrays."""
        self._flat_nodes = []
        self._flatten_post_order(self._root)

        for idx, node in enumerate(self._flat_nodes):
            node._flat_idx = idx
            if node.player >= 0 and len(node.children) > 0:
                child_keys = sorted(node.children.keys())
                node._child_keys = child_keys
                n_child = len(child_keys)
                node.regret = np.zeros((_N_HANDS, n_child), dtype=np.float32)
                node.strategy_sum = np.zeros((_N_HANDS, n_child), dtype=np.float32)
            else:
                node._child_keys = []

    def _flatten_post_order(self, node: SubgameNode) -> None:
        """Recursively flatten tree in post-order."""
        for key in sorted(node.children.keys()):
            self._flatten_post_order(node.children[key])
        self._flat_nodes.append(node)

    # ------------------------------------------------------------------
    # Terminal CFV computation (inline, uses cached equity)
    # ------------------------------------------------------------------

    def _terminal_cfv_exact(self, node: SubgameNode) -> tuple[np.ndarray, np.ndarray]:
        tt = node.terminal_type
        hero_pot = node.terminal_hero_pot
        opp_pot = node.terminal_opp_pot
        pot = node.pot
        
        cfv1 = np.zeros(_N_HANDS, dtype=np.float32)
        cfv2 = np.zeros(_N_HANDS, dtype=np.float32)
        
        if tt == _TERM_FOLD_P1:
            cfv1[:] = -hero_pot * node.opp_reach.sum()
            cfv2[:] = (pot - opp_pot) * node.hero_reach.sum()
        elif tt == _TERM_FOLD_P2:
            cfv1[:] = (pot - hero_pot) * node.opp_reach.sum()
            cfv2[:] = -opp_pot * node.hero_reach.sum()
        else:
            win_prob_hero = self._matchup_matrix @ node.opp_reach
            cfv1 = pot * win_prob_hero - hero_pot * node.opp_reach.sum()
            
            
            from nlhe.cfr.equity import _HAND_CONFLICT
            # Opponent equity is 1 - Hero equity, but only for non-conflicting hands.
            # M[I, J] is Hero equity. M.T[J, I] is Hero equity for Opp hand J vs Hero hand I.
            # We want Opp equity = 1 - M.T[J, I].
            # Valid reach for Opp hand J is sum_{I ~conflict} hero_reach[I].
            # win_prob_opp[J] = sum_{I ~conflict} (1 - M.T[J, I]) * hero_reach[I]
            # win_prob_opp[J] = valid_reach_sum[J] - (M.T @ hero_reach)[J]
            valid_reach_sum = (~_HAND_CONFLICT) @ node.hero_reach
            win_prob_opp = valid_reach_sum - (self._matchup_matrix.T @ node.hero_reach)

            cfv2 = pot * win_prob_opp - opp_pot * node.hero_reach.sum()

        cfv1[self._board_dead_mask] = 0.0
        cfv2[self._board_dead_mask] = 0.0
        return cfv1, cfv2

    def _cfr_iteration(self, t: int) -> None:
        # Top-down pass
        for node in reversed(self._flat_nodes):
            if node == self._root:
                node.hero_reach = self._hero_range.copy()
                node.opp_reach = self.opp_range.copy()
                
            if node.player == -1 or len(node.children) == 0:
                continue
                
            n_child = len(node._child_keys)
            strategy = _regret_match_per_hand(node.regret, n_child)
            # RNR blend removed for pure Nash
            node.current_strategy = strategy
            
            for a_idx, act_key in enumerate(node._child_keys):
                child = node.children[act_key]
                if node.player == 0:
                    child.hero_reach = node.hero_reach * strategy[:, a_idx]
                    child.opp_reach = node.opp_reach
                else:
                    child.hero_reach = node.hero_reach
                    child.opp_reach = node.opp_reach * strategy[:, a_idx]

        # Bottom-up pass
        node_cfv_p1 = {}
        node_cfv_p2 = {}

        for node in self._flat_nodes:
            if node.player == -1 or len(node.children) == 0:
                node_cfv_p1[node._flat_idx], node_cfv_p2[node._flat_idx] = self._terminal_cfv_exact(node)
                continue

            n_child = len(node._child_keys)
            strategy = node.current_strategy

            child_cfvs_p1 = np.empty((_N_HANDS, n_child), dtype=np.float32)
            child_cfvs_p2 = np.empty((_N_HANDS, n_child), dtype=np.float32)

            for a_idx, act_key in enumerate(node._child_keys):
                child = node.children[act_key]
                child_cfvs_p1[:, a_idx] = node_cfv_p1[child._flat_idx]
                child_cfvs_p2[:, a_idx] = node_cfv_p2[child._flat_idx]

            cfv_p1 = (strategy * child_cfvs_p1).sum(axis=1)
            cfv_p2 = (strategy * child_cfvs_p2).sum(axis=1)

            if node.player == 0:
                node.regret += child_cfvs_p1 - cfv_p1[:, None]
            else:
                node.regret += child_cfvs_p2 - cfv_p2[:, None]

            node_cfv_p1[node._flat_idx] = cfv_p1
            node_cfv_p2[node._flat_idx] = cfv_p2

            weight = ((t + 1) / (t + 2)) ** _DCFR_GAMMA if t > 0 else 1.0
            if node.player == 0:
                node.strategy_sum += weight * node.hero_reach[:, None] * strategy
            else:
                node.strategy_sum += weight * node.opp_reach[:, None] * strategy

            if t > 0:
                tp = t + 1
                pos_weight = (tp ** _DCFR_ALPHA) / (tp ** _DCFR_ALPHA + 1.0)
                pos_mask = node.regret > 0
                node.regret[pos_mask] *= pos_weight
                node.regret[~pos_mask] = 0.0
            else:
                np.maximum(node.regret, 0.0, out=node.regret)

    def _extract_hand_cfvs(self) -> np.ndarray:
        if self._root is None:
            return np.zeros(len(self._valid_actions), dtype=np.float32)

        root = self._root
        n_root_children = len(root._child_keys)

        action_cfvs = np.zeros(n_root_children, dtype=np.float32)
        for a_idx, act_key in enumerate(root._child_keys):
            child = root.children[act_key]
            action_cfvs[a_idx] = self._hero_cfv_at_node(child, self.opp_range)

        result = np.zeros(len(self._valid_actions), dtype=np.float32)
        key_to_idx = {k: i for i, k in enumerate(root._child_keys)}

        for va_idx, va_str in enumerate(self._valid_actions):
            act_int = _action_str_to_int(va_str)
            if act_int in key_to_idx:
                result[va_idx] = action_cfvs[key_to_idx[act_int]]
            else:
                result[va_idx] = -1e6

        return result

    def _hero_cfv_at_node(self, node: SubgameNode, opp_reaching: np.ndarray) -> float:
        total_opp = opp_reaching.sum()
        if total_opp < 1e-10:
            return 0.0

        if node.player == -1 or len(node.children) == 0:
            tt = node.terminal_type
            hero_pot = node.terminal_hero_pot
            pot = node.pot
            if tt == _TERM_FOLD_P1:
                return -hero_pot
            elif tt == _TERM_FOLD_P2:
                return pot - hero_pot
            else:
                M = self._matchup_matrix
                our_idx = HAND_TO_IDX.get(self.our_hole_idx)
                if our_idx is not None:
                    eq = (M[our_idx] * opp_reaching).sum() / total_opp
                else:
                    eq = 0.5
                return eq * pot - hero_pot

        if node.player == 0:
            best = -1e9
            for act_key in node._child_keys:
                child = node.children[act_key]
                v = self._hero_cfv_at_node(child, opp_reaching)
                if v > best:
                    best = v
            return best
        else:
            n_child = len(node._child_keys)
            if getattr(node, 'strategy_sum', None) is None:
                avg_strat = np.full((_N_HANDS, n_child), 1.0 / n_child, dtype=np.float32)
            else:
                totals = node.strategy_sum.sum(axis=1, keepdims=True)
                has_data = totals.ravel() > 0
                avg_strat = np.full((_N_HANDS, n_child), 1.0 / n_child, dtype=np.float32)
                avg_strat[has_data] = node.strategy_sum[has_data] / totals[has_data]

            weighted_cfv = 0.0
            for a_idx, act_key in enumerate(node._child_keys):
                child = node.children[act_key]
                child_reaching = opp_reaching * avg_strat[:, a_idx]
                if child_reaching.sum() < 1e-10:
                    continue
                weighted_cfv += self._hero_cfv_at_node(child, child_reaching)

            return weighted_cfv

    # ------------------------------------------------------------------
    # Root average strategy
    # ------------------------------------------------------------------

    def _get_root_avg_strategy(self) -> np.ndarray:
        """Get average strategy at the root for our specific hand."""
        if self._root is None or self._root.strategy_sum is None:
            n = len(self._valid_actions)
            return np.full(n, 1.0 / n, dtype=np.float32)

        our_hand_idx = HAND_TO_IDX.get(self.our_hole_idx, 0)
        s = self._root.strategy_sum[our_hand_idx, :]
        total = s.sum()
        if total > 0:
            root_strat = s / total
        else:
            root_strat = np.full(len(self._root._child_keys),
                                 1.0 / len(self._root._child_keys), dtype=np.float32)

        # Map root child keys to valid_actions ordering
        result = np.zeros(len(self._valid_actions), dtype=np.float32)
        key_to_idx = {k: i for i, k in enumerate(self._root._child_keys)}

        for va_idx, va_str in enumerate(self._valid_actions):
            act_int = _action_str_to_int(va_str)
            if act_int in key_to_idx:
                result[va_idx] = root_strat[key_to_idx[act_int]]

        total = result.sum()
        if total > 0:
            result /= total
        else:
            result[:] = 1.0 / len(self._valid_actions)

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Phantom range widening
    # ------------------------------------------------------------------

    def _build_phantom_range(self) -> np.ndarray:
        """Build a balanced hero range to prevent opponent exploitation.
        
        Narrows the range based on the pot size, assuming we play a solid
        strategy. This forces the opponent CFR nodes to play a balanced
        Nash response to our represented range, rather than treating us
        as a completely random player in 4-bet pots.
        """
        r = np.ones(_N_HANDS, dtype=np.float32)
        for c_idx in self._dead_idxs:
            r[HAND_CONTAINS_CARD[c_idx]] = 0.0

        pot = float(self.state.pot)
        
        # If the pot is significantly larger than the starting pot,
        # we have put chips in. Narrow the range.
        if pot > 400.0:
            strength = self._estimate_opp_hand_strength(self.state)
            
            # The larger the pot relative to the starting stack (20,000 chips), 
            # the tighter we are.
            commitment = pot / 20000.0 
            
            # Weight towards top hands if pot is big
            # power = 1.0 (mild) to 4.0 (very tight)
            power = min(4.0, 1.0 + commitment * 6.0)
            weights = np.power(strength, power)
            r *= weights

        total = r.sum()
        if total > 0:
            r /= total
        else:
            r[:] = 1.0 / _N_HANDS
            
        # Ensure our actual hand is in the range with at least a small epsilon
        # so CFR can compute a valid CFV for it.
        our_hand_idx = HAND_TO_IDX.get(self.our_hole_idx)
        if our_hand_idx is not None:
            r[our_hand_idx] = max(r[our_hand_idx], 1e-4)
            r /= r.sum()
            
        return r

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_initial_range(self, state: GameState) -> np.ndarray:
        """Build opponent range, narrowed when facing bets.

        A uniform range is used when there's no outstanding bet. When facing
        a bet, the range is weighted by hand strength (stronger hands are
        more likely to have bet/raised).
        """
        r = np.ones(_N_HANDS, dtype=np.float32)
        for c_idx in self._dead_idxs:
            r[HAND_CONTAINS_CARD[c_idx]] = 0.0

        # Narrow range when facing a bet: opponent has shown strength.
        # Use the opponent model's calibrated range weights instead of
        # aggressive power formula (which assumed opponent never bluffs).
        to_call = getattr(state, 'to_call', 0)
        pot = state.pot
        if to_call > 0 and pot > 0:
            bet_frac = to_call / max(pot - to_call, 1)
            if bet_frac > 0.15:
                strength = self._estimate_opp_hand_strength(state)
                # Opponent model weights: (0.3 + 0.7*s) for raises,
                # giving a floor of 0.3 even for the weakest hand.
                # Scale by bet_frac so larger bets narrow more.
                base_w = self._opp_model.get_range_weights(strength, 'RAISE_LARGE')
                # Blend: small bets -> mild narrowing, large bets -> full narrowing
                blend = min(bet_frac, 1.5) / 1.5  # 0 to 1
                weights = (1.0 - blend) + blend * base_w
                r *= weights

        total = r.sum()
        if total > 0.0:
            r /= total
        else:
            r[:] = 1.0 / _N_HANDS
        return r

    def _estimate_opp_hand_strength(self, state: GameState) -> np.ndarray:
        """Estimate relative hand strength for each opponent hand on the board.

        Flop/turn/river: actual board-aware hand evaluation via treys.
        Preflop: rank-based approximation.
        """
        from nlhe.cfr.equity import _ALL_TREYS_CARDS

        strength = np.zeros(_N_HANDS, dtype=np.float32)

        if len(state.board) >= 3:
            # Flop/turn/river: evaluate all hands against the board
            from treys import Card, Evaluator
            evaluator = Evaluator()
            board_treys = [Card.new(c) for c in state.board]

            scores = np.full(_N_HANDS, 99999, dtype=np.int32)
            for idx in range(_N_HANDS):
                if self._dead_mask[idx]:
                    continue
                c1, c2 = ALL_HANDS[idx]
                opp_treys = [_ALL_TREYS_CARDS[c1], _ALL_TREYS_CARDS[c2]]
                scores[idx] = evaluator.evaluate(board_treys, opp_treys)

            valid = scores < 99999
            if valid.sum() > 0:
                min_s = scores[valid].min()
                max_s = scores[valid].max()
                if max_s > min_s:
                    strength[valid] = 1.0 - (scores[valid].astype(float) - min_s) / (max_s - min_s)
                else:
                    strength[valid] = 0.5
            return strength

        # Preflop: use rank-based approximation
        for idx in range(_N_HANDS):
            if self._dead_mask[idx]:
                continue
            c1, c2 = ALL_HANDS[idx]
            r1, r2 = c1 // 4, c2 // 4
            s1, s2 = c1 % 4, c2 % 4
            high, low = max(r1, r2), min(r1, r2)
            s = (high + low) / 24.0
            if r1 == r2:
                s = 0.5 + r1 / 24.0
            if s1 == s2:
                s += 0.04
            if high - low <= 2:
                s += 0.02
            strength[idx] = min(1.0, s)

        return strength

    def _compute_dead_card_idxs(self, state: GameState) -> list[int]:
        """Return card indices for all known dead cards."""
        dead = []
        for c in self.our_hole:
            dead.append(card_str_to_idx(c))
        for c in state.board:
            dead.append(card_str_to_idx(c))
        return dead


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _action_str_to_int(action_str: str) -> int:
    """Map action string to abstract action int."""
    _mapping = {
        'FOLD': FOLD,
        'CALL': CALL,
        'RAISE_SMALL': RAISE_SMALL,
        'RAISE_LARGE': RAISE_LARGE,
        'RAISE_ALLIN': RAISE_ALLIN,
        'RAISE_OVERBET': RAISE_OVERBET,
    }
    return _mapping.get(action_str, CALL)


def _regret_match_per_hand(
    regret: np.ndarray, n_child: int
) -> np.ndarray:
    """Per-hand regret matching: (1326, n_child) -> (1326, n_child) strategy."""
    pos = np.maximum(regret, 0.0)
    totals = pos.sum(axis=1, keepdims=True)
    mask = (totals > 0).ravel()
    strategy = np.full_like(pos, 1.0 / n_child)
    if mask.any():
        strategy[mask] = pos[mask] / totals[mask]
    return strategy
