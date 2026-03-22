"""
GPU-accelerated Preflop CFR+ Solver — fully vectorised by tree level.

All 121 decision nodes are processed level-by-level (only ~9 levels) using
batched tensor gather/scatter. Zero Python loops over individual nodes
during solve(). Achieves full GPU saturation on T4/A100.

Same API as preflop_cfr.py.
"""
import math
import numpy as np
import os
import time as _time
import torch
from collections import deque
from itertools import combinations

from submission.card_utils import combo_index_5, NUM_CARDS, COMB
from submission.cfr.action_abstraction import (
    FOLD, CALL, RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN, N_ACTIONS
)

NUM_HANDS_5 = 80730  # C(27, 5)

_CACHED_ALL_HANDS = None
_CACHED_HAND_INDICES = None


def _get_hand_cache():
    global _CACHED_ALL_HANDS, _CACHED_HAND_INDICES
    if _CACHED_ALL_HANDS is None:
        _CACHED_ALL_HANDS = list(combinations(range(NUM_CARDS), 5))
        _CACHED_HAND_INDICES = np.array(
            [combo_index_5(h) for h in _CACHED_ALL_HANDS], dtype=np.int32
        )
    return _CACHED_ALL_HANDS, _CACHED_HAND_INDICES


class PreflopTreeNode:
    __slots__ = ['player', 'pot', 'ip_committed', 'oop_committed', 'raises',
                 'children', 'is_terminal', 'terminal_type',
                 'ip_history_idx', 'oop_history_idx', 'is_root']

    def __init__(self, player, pot, ip_committed, oop_committed, raises=0, is_root=False):
        self.player = player
        self.pot = pot
        self.ip_committed = ip_committed
        self.oop_committed = oop_committed
        self.raises = raises
        self.children = {}
        self.is_terminal = False
        self.terminal_type = None
        self.ip_history_idx = -1
        self.oop_history_idx = -1
        self.is_root = is_root


class PreflopCFR:
    RAISE_CAP = 3  # 4bet is deepest; 5bets are all-in at 50bb anyway
    STARTING_POT = 3
    IP_COMMITTED_START = 1
    OOP_COMMITTED_START = 2
    STACK = 200

    def __init__(self, match_margin: float = 0.0, hands_remaining: int = 500,
                 sigma_per_hand: float = 20.0, device: str = None):
        self.match_margin = match_margin
        self.hands_remaining = hands_remaining
        self.sigma_per_hand = sigma_per_hand

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        print(f"  PreflopCFR using device: {self._device}")

        # Build tree (Python objects)
        self._root = self._build_tree(
            player=0, pot=self.STARTING_POT,
            ip_committed=self.IP_COMMITTED_START,
            oop_committed=self.OOP_COMMITTED_START,
            raises=0, is_root=True,
        )

        self._ip_nodes = []
        self._oop_nodes = []
        self._bfs_index_nodes()

        # Hand cache & HP5
        self._all_hands, hand_indices_np = _get_hand_cache()
        self._hand_indices = torch.from_numpy(hand_indices_np).long().to(self._device)
        self._hp5 = self._load_hp5()
        if self._hp5 is not None:
            hp5_vals = self._hp5[self._hand_indices]
            self._oop_mean_hp5 = float(hp5_vals.mean())
            self._ip_mean_hp5 = self._oop_mean_hp5
        else:
            self._oop_mean_hp5 = 0.0
            self._ip_mean_hp5 = 0.0

        # Flatten tree into vectorised level-based structure
        self._vectorise_tree()
        self._iterations = 0

    # ------------------------------------------------------------------
    # Tree construction (same logic as CPU version)
    # ------------------------------------------------------------------

    def _build_tree(self, player, pot, ip_committed, oop_committed, raises, is_root=False):
        node = PreflopTreeNode(player, pot, ip_committed, oop_committed, raises, is_root)
        actions = self._valid_actions(player, raises, is_root)
        for a in actions:
            if a == FOLD:
                child = PreflopTreeNode(1 - player, pot, ip_committed, oop_committed, raises)
                child.is_terminal = True
                child.terminal_type = 'fold_ip' if player == 0 else 'fold_oop'
                node.children[a] = child
            elif a == CALL:
                new_ip = max(ip_committed, oop_committed) if player == 0 else ip_committed
                new_oop = max(ip_committed, oop_committed) if player == 1 else oop_committed
                new_pot = new_ip + new_oop
                if is_root:
                    # SB limp: BB still gets to act (check or raise)
                    child = self._build_tree(1 - player, new_pot, new_ip, new_oop, raises, False)
                else:
                    child = PreflopTreeNode(1 - player, new_pot, new_ip, new_oop, raises)
                    child.is_terminal = True
                    child.terminal_type = 'call'
                node.children[a] = child
            else:
                new_ip, new_oop = self._raise_amounts(a, player, ip_committed, oop_committed, raises)
                new_pot = new_ip + new_oop
                child = self._build_tree(1 - player, new_pot, new_ip, new_oop, raises + 1, False)
                node.children[a] = child
        return node

    def _valid_actions(self, player, raises, is_root):
        actions = [FOLD, CALL]  # CALL at root = SB limp
        if raises < self.RAISE_CAP:
            actions.extend([RAISE_SMALL, RAISE_LARGE, RAISE_ALLIN])
        return actions

    def _raise_amounts(self, action, player, ip_committed, oop_committed, raises):
        facing = max(ip_committed, oop_committed)
        if action == RAISE_SMALL:
            # ~3× facing: standard open / 3bet sizing
            total = max(facing * 3.0, 6.0) if raises == 0 else max(facing * 3.0, facing + 3.0)
        elif action == RAISE_LARGE:
            # ~10× facing: large open / big 3bet, converges to all-in at depth
            total = max(facing * 10.0, 16.0) if raises == 0 else max(facing * 10.0, facing + 10.0)
        else:
            total = float(self.STACK)
        total = min(total, float(self.STACK))
        return (total, oop_committed) if player == 0 else (ip_committed, total)

    def _bfs_index_nodes(self):
        queue = deque([self._root])
        seen = set()
        while queue:
            node = queue.popleft()
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)
            if node.is_terminal or not node.children:
                continue
            if node.player == 0:
                node.ip_history_idx = len(self._ip_nodes)
                self._ip_nodes.append(node)
            else:
                node.oop_history_idx = len(self._oop_nodes)
                self._oop_nodes.append(node)
            for a in sorted(node.children.keys()):
                queue.append(node.children[a])

    def _load_hp5(self):
        try:
            path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'tables', 'hand_potential_5.npy'
            ))
            if os.path.exists(path):
                arr = np.load(path)
                return torch.from_numpy(arr).float().to(self._device)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Vectorise tree into level-based tensor ops
    # ------------------------------------------------------------------

    def _vectorise_tree(self):
        N = NUM_HANDS_5
        dev = self._device
        n_ip = len(self._ip_nodes)
        n_oop = len(self._oop_nodes)

        # ── Assign compact IDs via BFS ───────────────────────────────────
        compact_id = {}
        decision_list = []
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
            compact_id[nid] = len(decision_list)
            decision_list.append(node)
            for a in sorted(node.children.keys()):
                queue.append(node.children[a])

        n_decision = len(decision_list)

        # ── Compute depths, parents, info indices ────────────────────────
        depth = {0: 0}
        parent_compact = [0] * n_decision
        parent_action = [0] * n_decision
        parent_info = [0] * n_decision
        node_info = [0] * n_decision
        node_player = [0] * n_decision

        for cid, node in enumerate(decision_list):
            node_player[cid] = node.player
            node_info[cid] = node.ip_history_idx if node.player == 0 else node.oop_history_idx

            for a in sorted(node.children.keys()):
                child = node.children[a]
                if not child.is_terminal:
                    child_cid = compact_id[id(child)]
                    depth[child_cid] = depth[cid] + 1
                    parent_compact[child_cid] = cid
                    parent_action[child_cid] = a
                    parent_info[child_cid] = node_info[cid]

        # ── Group by level ───────────────────────────────────────────────
        max_depth = max(depth.values()) if depth else 0
        levels_raw = [[] for _ in range(max_depth + 1)]
        for cid, d in depth.items():
            levels_raw[d].append(cid)

        self._n_levels = max_depth + 1
        self._levels = [torch.tensor(l, dtype=torch.long, device=dev) for l in levels_raw]
        self._level_player = [node_player[levels_raw[l][0]] for l in range(self._n_levels)]

        # ── Assign terminal slot IDs ─────────────────────────────────────
        terminal_list = []
        terminal_id_map = {}

        for cid, node in enumerate(decision_list):
            for a in sorted(node.children.keys()):
                child = node.children[a]
                if child.is_terminal:
                    tid = len(terminal_list)
                    terminal_list.append(child)
                    terminal_id_map[(cid, a)] = tid

        n_terminal = len(terminal_list)
        n_combined = n_decision + n_terminal + 1  # +1 sentinel (always 0)
        sentinel_idx = n_decision + n_terminal

        # ── Build child_combined_idx ─────────────────────────────────────
        child_combined_np = np.full((n_decision, N_ACTIONS), sentinel_idx, dtype=np.int64)
        action_valid_np = np.zeros((n_decision, N_ACTIONS), dtype=np.float32)

        for cid, node in enumerate(decision_list):
            for a in node.children:
                child = node.children[a]
                action_valid_np[cid, a] = 1.0
                if child.is_terminal:
                    child_combined_np[cid, a] = n_decision + terminal_id_map[(cid, a)]
                else:
                    child_combined_np[cid, a] = compact_id[id(child)]

        # ── Precompute terminal CFVs ─────────────────────────────────────
        sigma = self.sigma_per_hand * math.sqrt(max(1, self.hands_remaining))
        sqrt2 = math.sqrt(2.0)

        tcfv_p0 = torch.zeros(N, n_terminal, device=dev)
        tcfv_p1 = torch.zeros(N, n_terminal, device=dev)

        for tid, term_node in enumerate(terminal_list):
            tcfv_p0[:, tid] = self._compute_terminal_cfv(term_node, 0, sigma, sqrt2)
            tcfv_p1[:, tid] = self._compute_terminal_cfv(term_node, 1, sigma, sqrt2)

        # ── Store all tensors ────────────────────────────────────────────
        self._n_decision = n_decision
        self._n_terminal = n_terminal
        self._n_combined = n_combined
        self._sentinel_idx = sentinel_idx
        self._root_cid = compact_id[id(self._root)]

        self._child_combined_idx = torch.from_numpy(child_combined_np).to(dev)
        self._action_valid = torch.from_numpy(action_valid_np).to(dev)
        self._node_info_t = torch.tensor(node_info, dtype=torch.long, device=dev)
        self._parent_compact_t = torch.tensor(parent_compact, dtype=torch.long, device=dev)
        self._parent_action_t = torch.tensor(parent_action, dtype=torch.long, device=dev)
        self._parent_info_t = torch.tensor(parent_info, dtype=torch.long, device=dev)

        # Pre-build combined CFV templates (terminal CFVs pre-filled, rest zero)
        self._cfv_template_p0 = torch.zeros(N, n_combined, device=dev)
        self._cfv_template_p0[:, n_decision:n_decision + n_terminal] = tcfv_p0
        self._cfv_template_p1 = torch.zeros(N, n_combined, device=dev)
        self._cfv_template_p1[:, n_decision:n_decision + n_terminal] = tcfv_p1
        del tcfv_p0, tcfv_p1

        # ── Uniform strategies ───────────────────────────────────────────
        self._ip_uniform = torch.zeros(n_ip, N_ACTIONS, device=dev)
        for node in self._ip_nodes:
            for a in node.children:
                self._ip_uniform[node.ip_history_idx, a] = 1.0
        self._ip_uniform /= self._ip_uniform.sum(dim=1, keepdim=True).clamp(min=1.0)

        self._oop_uniform = torch.zeros(n_oop, N_ACTIONS, device=dev)
        for node in self._oop_nodes:
            for a in node.children:
                self._oop_uniform[node.oop_history_idx, a] = 1.0
        self._oop_uniform /= self._oop_uniform.sum(dim=1, keepdim=True).clamp(min=1.0)

        # ── CFR+ arrays ─────────────────────────────────────────────────
        self._ip_regret = torch.zeros(N, n_ip, N_ACTIONS, device=dev)
        self._ip_strat = torch.zeros(N, n_ip, N_ACTIONS, device=dev)
        self._oop_regret = torch.zeros(N, n_oop, N_ACTIONS, device=dev)
        self._oop_strat = torch.zeros(N, n_oop, N_ACTIONS, device=dev)

        # ── Pre-allocate working buffers ─────────────────────────────────
        self._ip_reach = torch.zeros(N, n_decision, device=dev)
        self._oop_reach = torch.zeros(N, n_decision, device=dev)

        # Pre-flatten child indices per level for the bottom-up gather
        self._level_flat_idx = []
        self._level_n_nodes = []
        for l in range(self._n_levels):
            nodes_t = self._levels[l]
            cidx = self._child_combined_idx[nodes_t]  # (n_nodes, 5)
            self._level_flat_idx.append(cidx.reshape(-1))  # (n_nodes * 5,)
            self._level_n_nodes.append(len(nodes_t))

        print(f"  Tree vectorised: {n_decision} decision, {n_terminal} terminal, "
              f"{self._n_levels} levels, {n_ip} IP + {n_oop} OOP")

    def _compute_terminal_cfv(self, node, for_player, sigma, sqrt2):
        N = NUM_HANDS_5
        dev = self._device

        if node.terminal_type == 'fold_oop':
            chips_ip = float(node.oop_committed)
            if for_player == 0:
                x1 = (self.match_margin + chips_ip) / (sigma * sqrt2)
                x2 = self.match_margin / (sigma * sqrt2)
            else:
                x1 = (-self.match_margin - chips_ip) / (sigma * sqrt2)
                x2 = (-self.match_margin) / (sigma * sqrt2)
            val = 0.5 * (math.erf(x1) - math.erf(x2))
            return torch.full((N,), val, dtype=torch.float32, device=dev)

        elif node.terminal_type == 'fold_ip':
            chips_oop = float(node.ip_committed)
            if for_player == 0:
                x1 = (self.match_margin - chips_oop) / (sigma * sqrt2)
                x2 = self.match_margin / (sigma * sqrt2)
            else:
                x1 = (-self.match_margin + chips_oop) / (sigma * sqrt2)
                x2 = (-self.match_margin) / (sigma * sqrt2)
            val = 0.5 * (math.erf(x1) - math.erf(x2))
            return torch.full((N,), val, dtype=torch.float32, device=dev)

        else:  # call / showdown
            if self._hp5 is None:
                return torch.zeros(N, dtype=torch.float32, device=dev)
            pot_half = node.pot / 2.0
            hp5_vals = self._hp5[self._hand_indices]
            if for_player == 0:
                chip_ev = (hp5_vals - self._oop_mean_hp5) * pot_half
                margin = self.match_margin
            else:
                chip_ev = (hp5_vals - self._ip_mean_hp5) * pot_half
                margin = -self.match_margin
            x1 = (margin + chip_ev) / (sigma * sqrt2)
            x2 = margin / (sigma * sqrt2)
            return 0.5 * (torch.erf(x1) - math.erf(x2))

    # ------------------------------------------------------------------
    # Vectorised CFR+ iteration — level-based, no per-node Python loops
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cfr_iteration(self):
        N = NUM_HANDS_5

        # Compute all strategies in bulk
        pos_ip = torch.clamp(self._ip_regret, min=0.0)
        tot_ip = pos_ip.sum(dim=2, keepdim=True)
        ip_strat = torch.where(tot_ip > 0, pos_ip / tot_ip.clamp(min=1e-30),
                               self._ip_uniform.unsqueeze(0))

        pos_oop = torch.clamp(self._oop_regret, min=0.0)
        tot_oop = pos_oop.sum(dim=2, keepdim=True)
        oop_strat = torch.where(tot_oop > 0, pos_oop / tot_oop.clamp(min=1e-30),
                                self._oop_uniform.unsqueeze(0))

        # Init CFV buffers for BOTH players (fused — avoids duplicate reach pass)
        cfv_p0 = self._cfv_template_p0.clone()
        cfv_p1 = self._cfv_template_p1.clone()

        # Init reaches ONCE (identical for both player perspectives)
        self._ip_reach.zero_()
        self._oop_reach.zero_()
        self._ip_reach[:, self._root_cid] = 1.0
        self._oop_reach[:, self._root_cid] = 1.0

        # ── Top-down: propagate reaches (SINGLE PASS) ───────────────
        for level in range(1, self._n_levels):
            nodes_t = self._levels[level]
            par_player = self._level_player[level - 1]

            p_compact = self._parent_compact_t[nodes_t]
            p_info = self._parent_info_t[nodes_t]
            p_act = self._parent_action_t[nodes_t]

            par_strats = ip_strat[:, p_info, :] if par_player == 0 else oop_strat[:, p_info, :]
            idx = p_act.unsqueeze(0).unsqueeze(2).expand(N, -1, 1)
            s_a = par_strats.gather(2, idx).squeeze(2)

            if par_player == 0:
                self._ip_reach[:, nodes_t] = self._ip_reach[:, p_compact] * s_a
                self._oop_reach[:, nodes_t] = self._oop_reach[:, p_compact]
            else:
                self._oop_reach[:, nodes_t] = self._oop_reach[:, p_compact] * s_a
                self._ip_reach[:, nodes_t] = self._ip_reach[:, p_compact]

        # ── Bottom-up: compute CFVs for BOTH players + update regrets ──
        for level in reversed(range(self._n_levels)):
            nodes_t = self._levels[level]
            player = self._level_player[level]
            n_nodes = self._level_n_nodes[level]

            flat_idx = self._level_flat_idx[level]
            child_p0 = cfv_p0[:, flat_idx].view(N, n_nodes, N_ACTIONS)
            child_p1 = cfv_p1[:, flat_idx].view(N, n_nodes, N_ACTIONS)

            info_ids = self._node_info_t[nodes_t]
            strats = ip_strat[:, info_ids, :] if player == 0 else oop_strat[:, info_ids, :]
            valid = self._action_valid[nodes_t].unsqueeze(0)

            sv = strats * valid
            node_p0 = (sv * child_p0).sum(dim=2)
            node_p1 = (sv * child_p1).sum(dim=2)
            cfv_p0[:, nodes_t] = node_p0
            cfv_p1[:, nodes_t] = node_p1

            # Update IP regrets at IP nodes, OOP regrets at OOP nodes
            if player == 0:
                regret_delta = (child_p0 - node_p0.unsqueeze(2)) * valid
                cf_w = self._oop_reach[:, nodes_t].unsqueeze(2)
                self._ip_regret[:, info_ids, :] = torch.clamp(
                    self._ip_regret[:, info_ids, :] + cf_w * regret_delta, min=0.0)
                self._ip_strat[:, info_ids, :] += \
                    self._ip_reach[:, nodes_t].unsqueeze(2) * strats
            else:
                regret_delta = (child_p1 - node_p1.unsqueeze(2)) * valid
                cf_w = self._ip_reach[:, nodes_t].unsqueeze(2)
                self._oop_regret[:, info_ids, :] = torch.clamp(
                    self._oop_regret[:, info_ids, :] + cf_w * regret_delta, min=0.0)
                self._oop_strat[:, info_ids, :] += \
                    self._oop_reach[:, nodes_t].unsqueeze(2) * strats

        self._iterations += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, n_iterations):
        t0 = _time.time()
        for i in range(n_iterations):
            self.cfr_iteration()
            if (i + 1) % 1000 == 0:
                elapsed = _time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_iterations - i - 1) / rate
                print(f"    iter {i+1}/{n_iterations} | {rate:.0f} it/s | "
                      f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    def get_ip_strategy(self):
        strat = self._ip_strat.cpu().numpy()
        totals = strat.sum(axis=2, keepdims=True)
        return np.where(totals > 0, strat / np.where(totals > 0, totals, 1.0),
                        1.0 / N_ACTIONS).astype(np.float32)

    def get_oop_strategy(self):
        strat = self._oop_strat.cpu().numpy()
        totals = strat.sum(axis=2, keepdims=True)
        return np.where(totals > 0, strat / np.where(totals > 0, totals, 1.0),
                        1.0 / N_ACTIONS).astype(np.float32)

    def query_ip(self, hand5, history_idx):
        h_idx = combo_index_5(tuple(sorted(hand5)))
        strat = self._ip_strat[h_idx, history_idx].cpu().numpy().copy()
        total = strat.sum()
        if total > 0:
            return strat / total
        return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS

    def query_oop(self, hand5, history_idx):
        h_idx = combo_index_5(tuple(sorted(hand5)))
        strat = self._oop_strat[h_idx, history_idx].cpu().numpy().copy()
        total = strat.sum()
        if total > 0:
            return strat / total
        return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
