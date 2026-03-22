"""
CounterfactualValueNet: 7-layer × 256 PReLU network for estimating
CFR counterfactual values at street boundaries.

Input (705 dims base): [p1_range(351), p2_range(351), pot_frac, margin_norm, hands_norm]
River variant (732 dims): base + board_onehot(27)
Output (702 dims raw): [p1_cfv(351), p2_cfv(351)] — zero-sum corrected

Zero-sum correction: error = (dot(p1_range, p1_cfv_raw) + dot(p2_range, p2_cfv_raw)) / 2
  p1_cfv = p1_cfv_raw - error
  p2_cfv = p2_cfv_raw - error

Values output as fractions of pot for generalization across stack sizes.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError, ValueError):
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass

N_HANDS = 351
BOARD_DIM = 27         # one-hot for which of the 27 cards are on the board
INPUT_DIM = 705        # 351 + 351 + 3 (flop/turn nets)
RIVER_INPUT_DIM = 732  # 705 + 27 board one-hot (river net)
OUTPUT_DIM = 702       # 351 + 351
HIDDEN_DIM = 256
N_LAYERS = 7


class CounterfactualValueNet(nn.Module):
    def __init__(self, input_size=INPUT_DIM):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        super().__init__()
        self._input_size = input_size
        layers = []
        in_dim = input_size
        for _ in range(N_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.PReLU())
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(HIDDEN_DIM, OUTPUT_DIM))
        self._net = nn.Sequential(*layers)

    def _zero_sum_correct(self, raw_out, p1_range, p2_range) -> tuple:
        """
        raw_out: (B, 702) — raw network output
        p1_range: (B, 351)
        p2_range: (B, 351)
        Returns (p1_cfv, p2_cfv) each (B, 351), zero-sum corrected.
        """
        p1_raw = raw_out[:, :N_HANDS]    # (B, 351)
        p2_raw = raw_out[:, N_HANDS:]    # (B, 351)
        v1 = (p1_range * p1_raw).sum(dim=1, keepdim=True)  # (B, 1)
        v2 = (p2_range * p2_raw).sum(dim=1, keepdim=True)  # (B, 1)
        error = (v1 + v2) / 2.0
        p1_cfv = p1_raw - error
        p2_cfv = p2_raw - error
        return p1_cfv, p2_cfv

    def forward(self, p1_range: np.ndarray, p2_range: np.ndarray,
                pot_fraction: float, match_margin_norm: float,
                hands_remaining_norm: float, board_cards=None) -> tuple:
        """
        Single-situation inference. All numpy in/out.
        board_cards: tuple/list of card indices (0-26) on the board. Required for river net.
        Returns (p1_cfv, p2_cfv) each (351,) float32 numpy.
        """
        parts = [p1_range, p2_range,
                 np.array([pot_fraction, match_margin_norm, hands_remaining_norm],
                          dtype=np.float32)]
        if self._input_size > INPUT_DIM and board_cards is not None:
            board_oh = np.zeros(BOARD_DIM, dtype=np.float32)
            for c in board_cards:
                board_oh[c] = 1.0
            parts.append(board_oh)
        x = np.concatenate(parts)
        result = self.forward_batch(x[np.newaxis])
        return result[0][0], result[1][0]

    def forward_batch(self, batch_inputs: np.ndarray) -> tuple:
        """
        Batched inference. batch_inputs shape (B, input_size).
        Returns (p1_cfv, p2_cfv) each (B, 351) float32 numpy.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        x = torch.from_numpy(batch_inputs.astype(np.float32))
        p1_range_t = x[:, :N_HANDS]
        p2_range_t = x[:, N_HANDS:2*N_HANDS]
        with torch.no_grad():
            raw = self._net(x)
        p1_cfv_t, p2_cfv_t = self._zero_sum_correct(raw, p1_range_t, p2_range_t)
        return p1_cfv_t.numpy().astype(np.float32), p2_cfv_t.numpy().astype(np.float32)


def load_value_net(path: str):
    """
    Load trained CounterfactualValueNet from .pt file.
    Auto-detects input size from state dict (705 for flop/turn, 732 for river).
    Returns None if file is missing (triggers HS2 fallback in solver).
    """
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(path):
        return None
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        # Detect input size from first Linear layer's weight shape
        input_size = sd['_net.0.weight'].shape[1]
        net = CounterfactualValueNet(input_size=input_size)
        net.load_state_dict(sd)
        net.eval()
        return net
    except Exception:
        return None
