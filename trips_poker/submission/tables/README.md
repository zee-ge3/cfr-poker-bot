# Tables

Small lookup tables tracked in git:

| File | Size | Description |
|------|------|-------------|
| `hand_ranks.npy` | 1.7 MB | Hand rank lookup for all 27-card deals |
| `hand_potential_5.npy` | 316 KB | 5-card hand potential estimates |
| `hand_strength_2.npy` | 1.5 KB | 2-card hand strength baseline |
| `value_net_flop.pt` | 3.0 MB | PyTorch value net weights (flop) |
| `value_net_turn.pt` | 3.0 MB | PyTorch value net weights (turn) |
| `value_net_river.pt` | 3.0 MB | PyTorch value net weights (river) |

The following files are **not tracked in git** due to size (62-124 MB each).
They are precomputed by `cfr/preflop_cfr.py` / `cfr/preflop_cfr_gpu.py`:

- `preflop_strategy_ip_neutral.npy`
- `preflop_strategy_ip_protection.npy`
- `preflop_strategy_oop_neutral.npy`
- `preflop_strategy_oop_protection.npy`
- `optimal_keep_table.npy` — optimal card-keep decisions for the Trips discard street
