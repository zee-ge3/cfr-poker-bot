"""Card encoding, combinatorial indexing, and hand classification for 27-card poker."""

NUM_RANKS = 9   # 2,3,4,5,6,7,8,9,A
NUM_SUITS = 3   # d,h,s
NUM_CARDS = 27

# ── Card encoding ────────────────────────────────────────────────────────────
# card_int = suit_index * 9 + rank_index
# rank_index: 0=2, 1=3, ..., 7=9, 8=Ace
# suit_index: 0=diamonds, 1=hearts, 2=spades

def rank(card: int) -> int:
    return card % NUM_RANKS

def suit(card: int) -> int:
    return card // NUM_RANKS

def is_suited(c1: int, c2: int) -> bool:
    return suit(c1) == suit(c2)

def is_paired(c1: int, c2: int) -> bool:
    return rank(c1) == rank(c2)

def is_connected(c1: int, c2: int) -> bool:
    r1, r2 = rank(c1), rank(c2)
    diff = abs(r1 - r2)
    if diff <= 2:
        return True
    # Ace (rank 8) wraps to low: A-2 (diff=8), A-3 (diff=7) are connected
    # because A-2-3-4-5 is a valid straight in standard poker
    # Also 9-A (diff=1) already handled above
    # In this 27-card deck, A can act as 10 for 6-7-8-9-A straights
    # So A connects with 7,8,9 (high) and 2,3 (low)
    if r1 == 8 or r2 == 8:  # one is Ace
        other = r2 if r1 == 8 else r1
        return other <= 1 or other >= 6  # 2,3 (low wrap) or 7,8,9 (high)
    return False

def is_high(card: int) -> bool:
    return rank(card) >= 6  # 8, 9, or Ace

# ── Combinatorial indexing ───────────────────────────────────────────────────
# Combinatorial number system: maps sorted k-card tuples to unique indices

# Precompute C(n, k) for n=0..27, k=0..7
COMB = [[0] * 8 for _ in range(28)]
for _n in range(28):
    COMB[_n][0] = 1
    for _k in range(1, min(_n + 1, 8)):
        COMB[_n][_k] = COMB[_n - 1][_k - 1] + COMB[_n - 1][_k]

def combo_index(cards: tuple, k: int) -> int:
    """Map sorted k-card combo to unique index in [0, C(27,k))."""
    s = sorted(cards)
    return sum(COMB[s[i]][i + 1] for i in range(k))

def combo_index_2(cards: tuple) -> int:
    return combo_index(cards, 2)

def combo_index_5(cards: tuple) -> int:
    return combo_index(cards, 5)

def combo_index_7(cards: tuple) -> int:
    return combo_index(cards, 7)

# ── Hand classification ──────────────────────────────────────────────────────
# WrappedEval score ranges (lower = stronger):

def classify_hand(score: int) -> str:
    if score <= 10:    return "straight_flush"
    if score <= 322:   return "full_house"
    if score <= 1599:  return "flush"
    if score <= 1609:  return "straight"
    if score <= 2467:  return "trips"
    if score <= 3325:  return "two_pair"
    if score <= 6185:  return "pair"
    return "high_card"
