"""
Offline Preflop CFR+ Solver.

Game: heads-up, SB posts 1, BB posts 2, starting pot=3.
SB acts first preflop. No limp option.
4-raise cap per street.
History is a BFS-indexed sequence of abstract actions over non-terminal decision nodes.

Abstract actions: FOLD=0, CALL=1, RAISE_SMALL=2, RAISE_LARGE=3, RAISE_ALLIN=4

Output: strategy[hand_idx, history_idx] -> float32[5] probability distribution
  - IP tables: shape (80730, 91, 5)
  - OOP tables: shape (80730, 30, 5)
"""
import math
import numpy as np
import os
from collections import deque
from itertools import combinations

from submission.card_utils import combo_index_5, NUM_CARDS, COMB
from submission.cfr.action_abstraction import (
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, N_ACTIONS
)
from submission.cfr.utility import tournament_utility

NUM_HANDS_5 = 80730  # C(27, 5)

# Expected BFS node counts (asserted at init)
EXPECTED_IP_NODES = 91
EXPECTED_OOP_NODES = 30

# Module-level cache — computed once, shared across all PreflopCFR instances
_CACHED_ALL_HANDS = None
_CACHED_HAND_INDICES = None


def _get_hand_cache():
    """Return (all_hands, hand_indices), computing once and caching at module level."""
    global _CACHED_ALL_HANDS, _CACHED_HAND_INDICES
    if _CACHED_ALL_HANDS is None:
        _CACHED_ALL_HANDS = list(combinations(range(NUM_CARDS), 5))
        _CACHED_HAND_INDICES = np.array(
            [combo_index_5(h) for h in _CACHED_ALL_HANDS], dtype=np.int32
        )
    return _CACHED_ALL_HANDS, _CACHED_HAND_INDICES


def _erf_approx_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized erf approximation (Abramowitz & Stegun 7.1.26, error < 1.5e-7).
    scipy is NOT available; this replaces scipy.special.erf.
    """
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * np.abs(x))
    t2, t3, t4, t5 = t*t, t*t*t, t*t*t*t, t*t*t*t*t
    poly = a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
    y = 1.0 - poly * np.exp(-(x * x))
    return np.where(x >= 0, y, -y)


class PreflopTreeNode:
    """Node in the preflop game tree."""
    __slots__ = ['player', 'pot', 'ip_committed', 'oop_committed', 'raises',
                 'children', 'is_terminal', 'terminal_type',
                 'ip_history_idx', 'oop_history_idx', 'is_root']

    def __init__(self, player, pot, ip_committed, oop_committed, raises=0, is_root=False):
        self.player = player          # 0=IP/SB, 1=OOP/BB
        self.pot = pot
        self.ip_committed = ip_committed
        self.oop_committed = oop_committed
        self.raises = raises           # raises so far this street
        self.children = {}             # action → PreflopTreeNode
        self.is_terminal = False
        self.terminal_type = None      # 'fold_ip', 'fold_oop', 'call'
        self.ip_history_idx = -1       # BFS index (IP decision nodes)
        self.oop_history_idx = -1      # BFS index (OOP decision nodes)
        self.is_root = is_root


class PreflopCFR:
    """
    CFR+ solver for preflop game tree.

    Builds the tree once, then iterates CFR+ until `solve()` is called.
    Output: strategy and average strategy arrays indexed by (hand_idx, history_idx).

    Tree structure:
      - SB (IP, player=0) acts first. No limp. Actions: FOLD, RS, RL, AI.
      - After a raise, opponent gets: FOLD, CALL, and raises (if cap not reached).
      - At raise cap (raises==RAISE_CAP), opponent gets only FOLD, CALL.
      - CALL always terminates the street (showdown / go to flop).
      - FOLD always terminates (folder loses their committed chips).

    BFS node counts: 91 IP nodes, 30 OOP nodes.
    """

    RAISE_CAP = 4
    STARTING_POT = 3
    IP_COMMITTED_START = 1   # SB posts 1
    OOP_COMMITTED_START = 2  # BB posts 2
    STACK = 200              # effective stack each player starts with

    def __init__(self, match_margin: float = 0.0, hands_remaining: int = 500,
                 sigma_per_hand: float = 20.0):
        self.match_margin = match_margin
        self.hands_remaining = hands_remaining
        self.sigma_per_hand = sigma_per_hand

        # Build tree
        self._root = self._build_tree(
            player=0,
            pot=self.STARTING_POT,
            ip_committed=self.IP_COMMITTED_START,
            oop_committed=self.OOP_COMMITTED_START,
            raises=0,
            is_root=True,
        )

        # BFS-index all non-terminal IP and OOP decision nodes
        self._ip_nodes = []   # list of PreflopTreeNode for IP, BFS order
        self._oop_nodes = []  # list of PreflopTreeNode for OOP, BFS order
        self._bfs_index_nodes()

        # Assert counts
        assert len(self._ip_nodes) == EXPECTED_IP_NODES, \
            f"Expected {EXPECTED_IP_NODES} IP nodes, got {len(self._ip_nodes)}"
        assert len(self._oop_nodes) == EXPECTED_OOP_NODES, \
            f"Expected {EXPECTED_OOP_NODES} OOP nodes, got {len(self._oop_nodes)}"

        # CFR+ arrays: regret_sum and strategy_sum per node per hand
        self._ip_regret  = np.zeros((NUM_HANDS_5, EXPECTED_IP_NODES,  N_ACTIONS), dtype=np.float32)
        self._ip_strat   = np.zeros((NUM_HANDS_5, EXPECTED_IP_NODES,  N_ACTIONS), dtype=np.float32)
        self._oop_regret = np.zeros((NUM_HANDS_5, EXPECTED_OOP_NODES, N_ACTIONS), dtype=np.float32)
        self._oop_strat  = np.zeros((NUM_HANDS_5, EXPECTED_OOP_NODES, N_ACTIONS), dtype=np.float32)

        # Cache all hands (module-level singleton — computed only once per process)
        self._all_hands, self._hand_indices = _get_hand_cache()

        # Load HAND_POTENTIAL_5 for leaf evaluation (may be None)
        self._hp5 = self._load_hp5()

        # Precompute mean HP5 for vectorized terminal evaluation
        # Both players face the same distribution at preflop (symmetric)
        if self._hp5 is not None:
            mean_hp5 = float(self._hp5[self._hand_indices].mean())
            self._oop_mean_hp5 = mean_hp5
            self._ip_mean_hp5 = mean_hp5
        else:
            self._oop_mean_hp5 = 0.0
            self._ip_mean_hp5 = 0.0

        self._iterations = 0

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, player, pot, ip_committed, oop_committed, raises, is_root=False):
        """
        Recursively build the preflop game tree.

        is_root=True  → SB's first action; CALL (limp) is excluded.
        """
        node = PreflopTreeNode(player, pot, ip_committed, oop_committed, raises, is_root)

        actions = self._valid_actions(player, raises, is_root)

        for a in actions:
            if a == FOLD:
                child = PreflopTreeNode(1 - player, pot, ip_committed, oop_committed, raises)
                child.is_terminal = True
                child.terminal_type = 'fold_ip' if player == 0 else 'fold_oop'
                node.children[a] = child

            elif a == CALL:
                # CALL ends the street — showdown / proceed to flop
                new_ip  = max(ip_committed, oop_committed) if player == 0 else ip_committed
                new_oop = max(ip_committed, oop_committed) if player == 1 else oop_committed
                new_pot = new_ip + new_oop
                child = PreflopTreeNode(1 - player, new_pot, new_ip, new_oop, raises)
                child.is_terminal = True
                child.terminal_type = 'call'
                node.children[a] = child

            else:  # RAISE_SMALL / RAISE_LARGE / RAISE_ALLIN
                new_ip_committed, new_oop_committed = self._raise_amounts(
                    a, player, ip_committed, oop_committed, raises
                )
                new_pot = new_ip_committed + new_oop_committed
                child = self._build_tree(
                    player=1 - player,
                    pot=new_pot,
                    ip_committed=new_ip_committed,
                    oop_committed=new_oop_committed,
                    raises=raises + 1,
                    is_root=False,
                )
                node.children[a] = child

        return node

    def _valid_actions(self, player, raises, is_root):
        """Return valid abstract actions."""
        actions = []
        # FOLD always available (even at root, SB can fold and lose the blind)
        actions.append(FOLD)
        # CALL available unless this is root (no limp for SB)
        if not is_root:
            actions.append(CALL)
        # Raises available if cap not reached
        if raises < self.RAISE_CAP:
            actions.extend([RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN])
        return actions

    def _raise_amounts(self, action, player, ip_committed, oop_committed, raises):
        """Compute new ip_committed and oop_committed after a raise."""
        facing = max(ip_committed, oop_committed)

        if action == RAISE_SMALL:
            if raises == 0:
                # First raise: open to 2.5bb minimum (facing=2)
                total = max(facing * 2.5, 6.0)
            else:
                total = max(facing * 2.5, facing + 2.0)
        elif action == RAISE_LARGE:
            if raises == 0:
                total = max(facing * 5.0, 10.0)
            else:
                total = max(facing * 5.0, facing + 4.0)
        else:  # RAISE_ALLIN
            total = float(self.STACK)

        total = min(total, float(self.STACK))

        if player == 0:
            return total, oop_committed
        else:
            return ip_committed, total

    # ------------------------------------------------------------------
    # BFS indexing
    # ------------------------------------------------------------------

    def _bfs_index_nodes(self):
        """BFS traversal to assign history indices to non-terminal IP and OOP nodes."""
        queue = deque([self._root])
        seen = set()

        while queue:
            node = queue.popleft()
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)

            if node.is_terminal:
                continue
            # Only index nodes that have non-terminal children to decide among
            if not node.children:
                continue

            if node.player == 0:
                node.ip_history_idx = len(self._ip_nodes)
                self._ip_nodes.append(node)
            else:
                node.oop_history_idx = len(self._oop_nodes)
                self._oop_nodes.append(node)

            for a in sorted(node.children.keys()):
                child = node.children[a]
                queue.append(child)

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def _get_current_strategy(self, hand_idx: int, node: PreflopTreeNode) -> np.ndarray:
        """
        Current strategy for this hand at this node via CFR+ regret matching.
        Returns a float32 array of shape (N_ACTIONS,).
        """
        if node.player == 0:
            h_idx = node.ip_history_idx
            regret = self._ip_regret[hand_idx, h_idx]
        else:
            h_idx = node.oop_history_idx
            regret = self._oop_regret[hand_idx, h_idx]

        positive = np.maximum(regret, 0.0)
        total = positive.sum()
        if total > 0:
            return positive / total

        n = len(node.children)
        strat = np.zeros(N_ACTIONS, dtype=np.float32)
        for a in node.children:
            strat[a] = 1.0 / n
        return strat

    # ------------------------------------------------------------------
    # Terminal utility
    # ------------------------------------------------------------------

    def _load_hp5(self):
        try:
            path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'tables', 'hand_potential_5.npy'
            ))
            if os.path.exists(path):
                return np.load(path)
        except Exception:
            pass
        return None

    def _terminal_utility(self, node: PreflopTreeNode, ip_hand_idx: int,
                          oop_weights: np.ndarray) -> float:
        """
        Compute IP's utility at a terminal node.

        For fold terminals: exact chip gain/loss.
        For call/showdown: chip_ev = HP5[ip] - E[HP5[oop]] under oop_weights.
        Blocking effects are ignored (documented approximation).
        """
        if node.terminal_type == 'fold_oop':
            chips_won = node.oop_committed
            return tournament_utility(chips_won, self.match_margin,
                                      self.hands_remaining, self.sigma_per_hand)
        elif node.terminal_type == 'fold_ip':
            chips_lost = node.ip_committed
            return tournament_utility(-chips_lost, self.match_margin,
                                      self.hands_remaining, self.sigma_per_hand)
        else:  # call / showdown
            if self._hp5 is None:
                return 0.0
            ip_hp = float(self._hp5[ip_hand_idx])
            oop_hp = float(np.dot(self._hp5[self._hand_indices], oop_weights))
            chip_ev = ip_hp - oop_hp
            return tournament_utility(chip_ev, self.match_margin,
                                      self.hands_remaining, self.sigma_per_hand)

    # ------------------------------------------------------------------
    # CFR+ traversal
    # ------------------------------------------------------------------

    def cfr_iteration(self) -> None:
        """
        Run one full CFR+ iteration — alternating IP and OOP traversals.

        Standard 2-player CFR: each iteration does TWO traversals —
        one updating IP regrets (with OOP reach as counterfactual weight),
        one updating OOP regrets (with IP reach as counterfactual weight).
        Previous version only did IP traversal, leaving OOP at uniform forever.
        """
        N = NUM_HANDS_5
        # Traversal 1: update IP regrets (OOP is the "opponent")
        self._traverse_vec(self._root,
                           ip_reach=np.ones(N, dtype=np.float64),
                           oop_reach=np.ones(N, dtype=np.float64),
                           update_player=0)
        # Traversal 2: update OOP regrets (IP is the "opponent")
        self._traverse_vec(self._root,
                           ip_reach=np.ones(N, dtype=np.float64),
                           oop_reach=np.ones(N, dtype=np.float64),
                           update_player=1)
        self._iterations += 1

    # ------------------------------------------------------------------
    # Vectorized traversal (all hands simultaneously)
    # ------------------------------------------------------------------

    def _get_strategy_vec(self, node: 'PreflopTreeNode') -> np.ndarray:
        """
        Regret-matched strategy for all hands at this node.
        Returns shape (N_HANDS_5, N_ACTIONS) float64.
        """
        N = NUM_HANDS_5
        if node.player == 0:
            regrets = self._ip_regret[:, node.ip_history_idx, :]   # (N, 5)
        else:
            regrets = self._oop_regret[:, node.oop_history_idx, :]  # (N, 5)

        positive = np.maximum(regrets, 0.0)
        totals = positive.sum(axis=1, keepdims=True)               # (N, 1)

        valid_mask = np.zeros(N_ACTIONS, dtype=np.float64)
        for a in node.children:
            valid_mask[a] = 1.0
        n_valid = max(1, valid_mask.sum())
        uniform = valid_mask / n_valid

        # Where total>0: regret-match; else: uniform over valid actions
        strategy = np.where(
            totals > 0,
            positive / np.where(totals > 0, totals, 1.0),
            uniform[np.newaxis, :]
        )
        return strategy  # (N, 5) float64

    def _terminal_cfv_vec(self, node: 'PreflopTreeNode',
                          for_player: int) -> np.ndarray:
        """
        Vectorized terminal utility for all hands, from for_player's perspective.
        Returns shape (N_HANDS_5,) float64.

        for_player: 0=IP, 1=OOP. Fold and showdown values are negated for OOP.
        """
        N = NUM_HANDS_5
        if node.terminal_type == 'fold_oop':
            # OOP folded → IP wins OOP's committed chips
            chips_ip = float(node.oop_committed)
            u_ip = tournament_utility(chips_ip, self.match_margin,
                                      self.hands_remaining, self.sigma_per_hand)
            u_oop = tournament_utility(-chips_ip, -self.match_margin,
                                       self.hands_remaining, self.sigma_per_hand)
            return np.full(N, u_ip if for_player == 0 else u_oop, dtype=np.float64)

        elif node.terminal_type == 'fold_ip':
            # IP folded → OOP wins IP's committed chips
            chips_oop = float(node.ip_committed)
            u_ip = tournament_utility(-chips_oop, self.match_margin,
                                      self.hands_remaining, self.sigma_per_hand)
            u_oop = tournament_utility(chips_oop, -self.match_margin,
                                       self.hands_remaining, self.sigma_per_hand)
            return np.full(N, u_ip if for_player == 0 else u_oop, dtype=np.float64)

        else:  # call / showdown
            if self._hp5 is None:
                return np.zeros(N, dtype=np.float64)
            # HP5 is in [0,1] range (equity). Convert to chip EV.
            # At a "call" terminal, each player has committed some chips.
            # IP's chip_ev ≈ (HP5[ip] - HP5_mean) * pot_half
            # Scale by committed chips to make fold/call/raise decisions meaningful.
            pot_half = node.pot / 2.0
            hp5_vals = self._hp5[self._hand_indices].astype(np.float64)

            if for_player == 0:
                chip_ev = (hp5_vals - self._oop_mean_hp5) * pot_half
                margin = self.match_margin
            else:
                # OOP's perspective: their hand is strong when HP5 is high,
                # but they want IP's HP5 to be low
                chip_ev = (hp5_vals - self._ip_mean_hp5) * pot_half
                margin = -self.match_margin

            sigma = self.sigma_per_hand * math.sqrt(max(1, self.hands_remaining))
            sqrt2 = math.sqrt(2.0)
            x1 = (margin + chip_ev) / (sigma * sqrt2)
            x2 = margin / (sigma * sqrt2)
            erf_x1 = _erf_approx_vec(x1)
            erf_x2 = math.erf(x2)
            return 0.5 * (erf_x1 - erf_x2)

    def _traverse_vec(self, node: 'PreflopTreeNode',
                      ip_reach: np.ndarray,
                      oop_reach: np.ndarray,
                      update_player: int) -> np.ndarray:
        """
        Vectorized CFR+ traversal for one player. Processes all hands simultaneously.

        Standard 2-player CFR: we traverse from update_player's perspective.
        The counterfactual weight for regret updates is the OTHER player's reach.

        ip_reach: (N,) float64 — IP reach probability per hand
        oop_reach: (N,) float64 — OOP reach probability per hand
        update_player: 0=update IP regrets, 1=update OOP regrets

        Returns cfv: (N,) float64 — counterfactual value per hand for update_player.
        """
        if node.is_terminal:
            return self._terminal_cfv_vec(node, for_player=update_player)

        strategy = self._get_strategy_vec(node)  # (N, 5)
        action_cfvs = {}

        for a, child in node.children.items():
            if node.player == 0:  # IP acts
                new_ip_reach = ip_reach * strategy[:, a]
                child_cfv = self._traverse_vec(child, new_ip_reach, oop_reach,
                                               update_player)
            else:  # OOP acts
                new_oop_reach = oop_reach * strategy[:, a]
                child_cfv = self._traverse_vec(child, ip_reach, new_oop_reach,
                                               update_player)
            action_cfvs[a] = child_cfv  # (N,)

        # Node CFV: weighted sum over actions
        N = len(ip_reach)
        node_cfv = np.zeros(N, dtype=np.float64)
        for a, cfv in action_cfvs.items():
            node_cfv += strategy[:, a] * cfv

        # CFR+ regret update: only for the update_player at their decision nodes
        if node.player == update_player:
            # Counterfactual weight = opponent's reach
            cf_weight = oop_reach if update_player == 0 else ip_reach

            if update_player == 0:
                h_idx = node.ip_history_idx
                for a, cfv in action_cfvs.items():
                    self._ip_regret[:, h_idx, a] = np.maximum(
                        0.0,
                        self._ip_regret[:, h_idx, a] + cf_weight * (cfv - node_cfv)
                    )
                self._ip_strat[:, h_idx, :] += ip_reach[:, np.newaxis] * strategy
            else:
                h_idx = node.oop_history_idx
                for a, cfv in action_cfvs.items():
                    self._oop_regret[:, h_idx, a] = np.maximum(
                        0.0,
                        self._oop_regret[:, h_idx, a] + cf_weight * (cfv - node_cfv)
                    )
                self._oop_strat[:, h_idx, :] += oop_reach[:, np.newaxis] * strategy

        return node_cfv

    def cfr_iteration_sequential(self) -> None:
        """Per-hand sequential CFR+ (slow; kept for debugging/validation)."""
        n_hands = len(self._all_hands)
        oop_weights = np.ones(n_hands, dtype=np.float32) / n_hands
        for ip_idx in range(n_hands):
            ip_hand_idx = int(self._hand_indices[ip_idx])
            self._traverse(self._root, ip_hand_idx, oop_weights,
                           reach_ip=1.0, reach_oop=1.0)
        self._iterations += 1

    def _traverse(self, node: PreflopTreeNode, ip_hand_idx: int,
                  oop_weights: np.ndarray, reach_ip: float, reach_oop: float) -> float:
        """
        Recursive CFR+ traversal.  Returns IP's counterfactual value.
        """
        if node.is_terminal:
            return self._terminal_utility(node, ip_hand_idx, oop_weights)

        strategy = self._get_current_strategy(ip_hand_idx, node)
        action_values = {}

        for a, child in node.children.items():
            if node.player == 0:  # IP acts
                val = self._traverse(child, ip_hand_idx, oop_weights,
                                     reach_ip * strategy[a], reach_oop)
            else:                 # OOP acts
                val = self._traverse(child, ip_hand_idx, oop_weights,
                                     reach_ip, reach_oop * strategy[a])
            action_values[a] = val

        node_value = sum(strategy[a] * v for a, v in action_values.items())

        # CFR+ regret update
        if node.player == 0:
            h_idx = node.ip_history_idx
            for a, v in action_values.items():
                self._ip_regret[ip_hand_idx, h_idx, a] = max(
                    0.0,
                    self._ip_regret[ip_hand_idx, h_idx, a] + reach_oop * (v - node_value)
                )
            self._ip_strat[ip_hand_idx, h_idx] += reach_ip * strategy
        else:
            # OOP update: accumulate strategy sum (regret would require iterating OOP hands)
            h_idx = node.oop_history_idx
            self._oop_strat[ip_hand_idx, h_idx] += reach_oop * strategy

        return node_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, n_iterations: int) -> None:
        """Run n_iterations of CFR+."""
        for _ in range(n_iterations):
            self.cfr_iteration()

    def get_ip_strategy(self) -> np.ndarray:
        """Return average IP strategy. Shape: (80730, 91, 5).

        Unvisited hand/node entries (sum==0) fall back to uniform over
        the actions available at that node.  For simplicity we use uniform
        over all N_ACTIONS; the caller should only query nodes reachable
        for the given hand.
        """
        totals = self._ip_strat.sum(axis=2, keepdims=True)  # (H, N, 1)
        zero_mask = (totals == 0.0).squeeze(axis=2)          # (H, N) bool
        strat = np.where(totals > 0, self._ip_strat / np.where(totals > 0, totals, 1.0),
                         1.0 / N_ACTIONS)
        return strat.astype(np.float32)

    def get_oop_strategy(self) -> np.ndarray:
        """Return average OOP strategy. Shape: (80730, 30, 5).

        Same fallback as get_ip_strategy for unvisited entries.
        """
        totals = self._oop_strat.sum(axis=2, keepdims=True)
        strat = np.where(totals > 0, self._oop_strat / np.where(totals > 0, totals, 1.0),
                         1.0 / N_ACTIONS)
        return strat.astype(np.float32)

    def load_strategy(self, ip_path: str, oop_path: str) -> None:
        """Load precomputed average strategies from .npy files."""
        self._ip_strat = np.load(ip_path)
        self._oop_strat = np.load(oop_path)

    def query_ip(self, hand5: tuple, history_idx: int) -> np.ndarray:
        """
        Query average IP strategy for a 5-card hand at a given history node index.
        Returns a probability distribution over N_ACTIONS.
        """
        h_idx = combo_index_5(tuple(sorted(hand5)))
        strat = self._ip_strat[h_idx, history_idx].copy()
        total = strat.sum()
        if total > 0:
            return strat / total
        return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS

    def query_oop(self, hand5: tuple, history_idx: int) -> np.ndarray:
        """Query average OOP strategy for a 5-card hand at a given history node index."""
        h_idx = combo_index_5(tuple(sorted(hand5)))
        strat = self._oop_strat[h_idx, history_idx].copy()
        total = strat.sum()
        if total > 0:
            return strat / total
        return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
