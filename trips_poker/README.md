# Trips Poker — CMU AI Poker Tournament Submission

This is the original competition code submitted to the CMU AI Poker Tournament.
The tournament used a custom 27-card "Trips Poker" variant (each player holds 3 cards,
discards one after the flop, standard 5-card board otherwise).

## Structure

```
submission/
  player.py                   # Main bot entry point (registered as the agent)
  cfr/
    subgame_cfr.py            # CFR-D depth-limited subgame solver (core algorithm)
    preflop_cfr.py            # CFR+ preflop strategy computation
    value_net.py              # Neural value function for leaf node estimation
    action_abstraction.py     # Bet-size bucketing
    variance_control.py       # Variance reduction helpers
  strategy.py                 # Strategy selection and blending
  equity.py                   # Hand equity calculator (Monte Carlo + exact)
  card_utils.py               # 27-card deck utilities
  opponent.py                 # Bayesian opponent range model
  tables/                     # Precomputed lookup tables (see tables/README.md)
    value_net_flop.pt         # Neural value net weights
    value_net_turn.pt
    value_net_river.pt
    hand_ranks.npy
    hand_potential_5.npy
    hand_strength_2.npy
```

## Key Algorithms

**CFR-D (Counterfactual Regret Decomposition):** Depth-limited subgame solving.
Rather than solving the full game tree (intractable), we solve a truncated subgame
with a learned value function at the leaf nodes. This trades exact GTO for a
practical per-hand solve within a 1-2 second time budget.

**CFR+ preflop tables:** Preflop strategy is precomputed offline using CFR+ (regret
matching with floor at zero, linear time weighting). Six tables cover IP/OOP positions
under neutral, protection, and pressure game states.

**Neural value nets:** Three PyTorch networks (flop/turn/river) estimate hand equity
from features including hole cards, board texture, pot size, and stack depth. Trained
via self-play data generation (`generate_value_data.py`).

**Opponent modeling:** Bayesian hand range updates based on observed actions, using
hand-strength percentile as a simplified type signal.

## Relationship to NLHE Portfolio Project

After the competition, the CFR-D architecture was generalized to standard
Heads-Up No-Limit Texas Hold'em. That work lives in `../nlhe/`. The core
solver ideas (depth-limited CFR, value network leaf estimation, opponent range
updating) are the same — the main differences are the deck size (52 vs 27 cards),
the hand evaluation logic, and the absence of the discard mechanic.
