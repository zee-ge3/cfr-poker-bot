"""
Streamlit demo: Heads-Up NLHE vs CFR-D Bot

Tab 1: Play vs Bot
Tab 2: Strategy Explorer
"""
import streamlit as st
import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from nlhe.game import NLHEGame, GameState, STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER
from nlhe.bot import Bot

st.set_page_config(page_title="NLHE CFR-D Bot", layout="centered")

SUIT_SYMBOLS = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
SUIT_COLORS  = {'s': '#000000', 'h': '#cc0000', 'd': '#cc0000', 'c': '#000000'}

STREET_NAMES = {
    STREET_PREFLOP: 'Preflop',
    STREET_FLOP:    'Flop',
    STREET_TURN:    'Turn',
    STREET_RIVER:   'River',
}

def fmt_card(c: str) -> str:
    """Format a treys card string as colored HTML."""
    rank, suit = c[0], c[1]
    sym = SUIT_SYMBOLS[suit]
    color = SUIT_COLORS[suit]
    return f'<span style="color:{color}; font-size:1.4em; font-weight:bold;">{rank}{sym}</span>'

def fmt_hand(cards: list) -> str:
    return ' '.join(fmt_card(c) for c in cards)


def run_bot_with_timeout(bot: Bot, state: GameState, timeout: float = 8.0):
    """Run bot.decide in a thread with a hard timeout. Returns action or fallback."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(bot.decide, state)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return 'CALL' if 'CALL' in state.valid_actions else state.valid_actions[0]


def init_session():
    if 'game' not in st.session_state:
        st.session_state.game = NLHEGame(starting_stack=100)
        st.session_state.bot = None
        st.session_state.hand_started = False
        st.session_state.hand_over = False
        st.session_state.result = None
        st.session_state.human_chips = 1000
        st.session_state.bot_chips = 1000
        st.session_state.log = []
        st.session_state.state = None
        st.session_state.waiting_for_bot = False
        st.session_state.human_position = 0


def play_tab():
    st.header("Play vs CFR-D Bot")
    st.caption("Heads-up NLHE. Starting stack: 100 chips.")

    init_session()
    game: NLHEGame = st.session_state.game

    col1, col2 = st.columns(2)
    col1.metric("Your chips", st.session_state.human_chips)
    col2.metric("Bot chips", st.session_state.bot_chips)

    if not st.session_state.hand_started:
        if st.button("Deal New Hand"):
            pos = st.session_state.human_position
            gs = game.new_hand(human_position=pos)

            bot = Bot(budget_seconds=6.0)
            bot.new_hand(hole_cards=gs.our_hole[:], position=1 - pos)
            st.session_state.bot = bot

            st.session_state.hand_started = True
            st.session_state.hand_over = False
            st.session_state.result = None
            st.session_state.state = gs
            st.session_state.log = [f"New hand dealt. You are {'SB' if pos == 0 else 'BB'}."]
            st.session_state.human_position = 1 - pos
            st.rerun()
        return

    gs: GameState = st.session_state.state

    st.markdown("**Board:** " + (fmt_hand(gs.board) if gs.board else "_No board yet_"),
                unsafe_allow_html=True)
    st.markdown("**Your hand:** " + fmt_hand(gs.our_hole), unsafe_allow_html=True)
    st.markdown(f"**Pot:** {gs.pot} chips &nbsp;|&nbsp; **Stack:** {gs.our_stack}",
                unsafe_allow_html=True)

    if st.session_state.hand_over:
        result = st.session_state.result
        if result:
            winner = result.get('winner', '')
            winnings = result.get('winnings', 0)
            bot_hole = result.get('bot_hole', [])
            reason = result.get('reason', '')
            if winner == 'human':
                st.success(f"You win {winnings} chips! ({reason})")
                st.session_state.human_chips += winnings // 2
                st.session_state.bot_chips -= winnings // 2
            elif winner == 'bot':
                st.error(f"Bot wins {winnings} chips. ({reason})")
                st.session_state.human_chips -= winnings // 2
                st.session_state.bot_chips += winnings // 2
            else:
                st.info("Tie!")
            if bot_hole:
                st.markdown("**Bot's hand:** " + fmt_hand(bot_hole), unsafe_allow_html=True)
        if st.button("Next Hand"):
            st.session_state.hand_started = False
            st.session_state.hand_over = False
            st.session_state.result = None
            st.session_state.state = None
            st.rerun()
        return

    st.markdown("---")
    st.markdown(f"**Street:** {STREET_NAMES.get(gs.street, gs.street)}")
    st.markdown("**Your action:**")
    cols = st.columns(len(gs.valid_actions))
    chosen_action = None
    for i, action in enumerate(gs.valid_actions):
        if cols[i].button(action, key=f"action_{action}_{gs.pot}"):
            chosen_action = action

    if chosen_action:
        result = game.human_action(chosen_action)
        log_msg = f"You: {chosen_action}"
        st.session_state.log.append(log_msg)

        if result['hand_over']:
            st.session_state.hand_over = True
            st.session_state.result = result
            st.session_state.state = result.get('state', gs)
            st.rerun()
        else:
            new_state = result.get('state', gs)
            st.session_state.state = new_state

            bot: Bot = st.session_state.bot
            bot.observe_action(chosen_action)
            with st.spinner("Bot thinking..."):
                bot_action = run_bot_with_timeout(bot, new_state, timeout=8.0)

            st.session_state.log.append(f"Bot: {bot_action}")
            bot_result = game.bot_action(bot_action)

            if bot_result['hand_over']:
                st.session_state.hand_over = True
                st.session_state.result = bot_result
            else:
                st.session_state.state = bot_result.get('state', new_state)

            st.rerun()

    if st.session_state.log:
        with st.expander("Hand log"):
            for entry in st.session_state.log:
                st.text(entry)


tab1, tab2 = st.tabs(["Play vs Bot", "Strategy Explorer"])

with tab1:
    play_tab()

with tab2:
    st.header("Strategy Explorer")
    st.caption("Select a scenario to see how the CFR-D solver reasons about it.")

    RANKS = list('23456789TJQKA')
    SUITS = ['c', 'd', 'h', 's']
    ALL_CARD_STRS = [r + s for r in RANKS for s in SUITS]

    col1, col2 = st.columns(2)
    hole1 = col1.selectbox("Your card 1", ALL_CARD_STRS, index=ALL_CARD_STRS.index('As'))
    hole2 = col2.selectbox("Your card 2", ALL_CARD_STRS, index=ALL_CARD_STRS.index('Kd'))

    board_input = st.multiselect("Board cards (0 for preflop, 3 for flop, 4 for turn, 5 for river)",
                                  ALL_CARD_STRS, max_selections=5)

    pot = st.slider("Pot size (chips)", 2, 200, 20)
    stack = st.slider("Stack depth (chips)", 1, 200, 80)
    position = st.radio("Your position", [0, 1], format_func=lambda x: "SB (BTN)" if x == 0 else "BB")

    if st.button("Solve"):
        our_hole = [hole1, hole2]
        board = list(board_input)

        if not board:
            from nlhe.cfr.preflop import PreflopTable
            from nlhe.cfr.equity import card_str_to_idx

            try:
                table = PreflopTable()
                idxs = tuple(sorted(card_str_to_idx(c) for c in our_hole))
                probs = table.lookup(idxs, position)
                action_names = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN']

                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.bar(action_names, probs, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
                ax.set_ylabel("Probability")
                ax.set_title(f"Preflop strategy: {hole1} {hole2} (from CFR+ table)")
                ax.set_ylim(0, 1)
                for bar, p in zip(bars, probs):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{p:.2f}', ha='center', fontsize=9)
                st.pyplot(fig)
                plt.close(fig)
            except FileNotFoundError:
                st.error("Preflop table not found. Run: python -m nlhe.cfr.preflop_compute")
        else:
            from nlhe.cfr.solver import NLHESolver

            street_map = {3: STREET_FLOP, 4: STREET_TURN, 5: STREET_RIVER}
            street = street_map.get(len(board), STREET_FLOP)
            valid_actions = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE', 'RAISE_ALLIN']

            state = GameState(
                street=street, pot=pot, our_stack=stack, opp_stack=stack,
                our_hole=our_hole, opp_hole=[], board=board,
                valid_actions=valid_actions, position=position, street_bets=[],
            )

            with st.spinner("Solving (up to 10s)..."):
                solver = NLHESolver(state, our_hole=our_hole, budget_seconds=10.0)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(solver.solve)
                    try:
                        chosen = future.result(timeout=10.0)
                    except concurrent.futures.TimeoutError:
                        chosen = 'CALL'

            probs = solver._avg_strategy if solver._avg_strategy is not None \
                    else np.ones(len(valid_actions)) / len(valid_actions)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            ax = axes[0]
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            ax.bar(valid_actions, probs, color=colors[:len(probs)])
            ax.set_ylabel("Probability")
            ax.set_title(f"CFR-D action distribution\n({hole1} {hole2} | board: {' '.join(board)})")
            ax.set_ylim(0, 1)
            for i, (v, p) in enumerate(zip(valid_actions, probs)):
                ax.text(i, p + 0.01, f'{p:.2f}', ha='center', fontsize=9)

            ax2 = axes[1]
            range_13x13 = np.zeros((13, 13))
            from nlhe.cfr.abstraction import ALL_HANDS
            opp_range = solver.opp_range
            for h_idx, (c1, c2) in enumerate(ALL_HANDS):
                r1, r2 = c1 // 4, c2 // 4
                range_13x13[12 - max(r1, r2)][12 - min(r1, r2)] += opp_range[h_idx]
            range_13x13 /= max(range_13x13.max(), 1e-10)
            rank_labels = list('AKQJT98765432')
            sns.heatmap(range_13x13, ax=ax2, cmap='Blues', vmin=0, vmax=1,
                        xticklabels=rank_labels, yticklabels=rank_labels, cbar=True)
            ax2.set_title("Opponent range (relative weight)")

            st.pyplot(fig)
            plt.close(fig)
            st.info(f"CFR-D recommends: **{chosen}**")

            st.markdown("**Estimated EV per action:**")
            from nlhe.cfr.equity import equity_mc
            ev_data = {}
            for a in valid_actions:
                if a == 'FOLD':
                    ev_data[a] = 0
                else:
                    eq = equity_mc(our_hole, board, n_samples=500)
                    ev_data[a] = round((eq - 0.5) * pot, 1)
            st.table(ev_data)
