"""
FastAPI poker demo — Heads-Up NLHE vs CFR-D Bot.

Routes:
  GET  /         full page
  GET  /game     game panel HTML (HTMX partial)
  POST /deal     deal new hand
  POST /action   process human action
  POST /solve    strategy explorer
"""
from __future__ import annotations

import concurrent.futures
import uuid
from typing import Optional

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import numpy as np

from nlhe.game import NLHEGame, GameState, STREET_NAMES
from nlhe.bot import Bot
from nlhe.cfr.equity import card_str_to_idx

app = FastAPI()

BASE_DIR = __file__[:__file__.rfind("/")]
templates = Jinja2Templates(directory=BASE_DIR + "/templates")

# ---------------------------------------------------------------------------
# Card rendering helpers
# ---------------------------------------------------------------------------

SUIT_SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLOR = {"s": "black", "h": "red", "d": "red", "c": "black"}

RANKS = list("23456789TJQKA")
SUITS = ["c", "d", "h", "s"]
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


def card_html(c: str) -> str:
    rank, suit = c[0], c[1]
    color = SUIT_COLOR[suit]
    sym = SUIT_SYM[suit]
    return f'<span class="card {color}">{rank}{sym}</span>'


def hand_html(cards: list[str]) -> str:
    return " ".join(card_html(c) for c in cards)


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
SESSION_COOKIE = "nlhe_session"


def get_session(request: Request) -> dict:
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and sid in _sessions:
        return _sessions[sid]
    return {}


def new_session() -> tuple[str, dict]:
    sid = str(uuid.uuid4())
    data: dict = {
        "game": NLHEGame(starting_stack=100),
        "bot": None,
        "state": None,
        "hand_started": False,
        "hand_over": False,
        "result": None,
        "human_chips": 1000,
        "bot_chips": 1000,
        "log": [],
        "human_position": 0,
    }
    _sessions[sid] = data
    return sid, data


def set_session_cookie(response: Response, sid: str) -> None:
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")


# ---------------------------------------------------------------------------
# Bot timeout wrapper
# ---------------------------------------------------------------------------

def bot_decide(bot: Bot, state: GameState, timeout: float = 8.0) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(bot.decide, state)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "CALL" if "CALL" in state.valid_actions else state.valid_actions[0]


# ---------------------------------------------------------------------------
# HTML partials
# ---------------------------------------------------------------------------

def render_game_panel(sess: dict) -> str:
    chips_bar = f"""
    <div class="chips-bar">
      <div class="chip-stat"><div class="val">{sess['human_chips']}</div><div class="lbl">Your chips</div></div>
      <div class="chip-stat"><div class="val">{sess['bot_chips']}</div><div class="lbl">Bot chips</div></div>
    </div>"""

    if not sess["hand_started"]:
        return chips_bar + """
        <form hx-post="/deal" hx-target="#game-area" hx-swap="innerHTML">
          <button type="submit" class="btn btn-primary">Deal New Hand</button>
        </form>"""

    gs: GameState = sess["state"]

    board_section = f"""
    <div class="cards-row">
      <div class="label">Board</div>
      {hand_html(gs.board) if gs.board else '<span style="color:#bbb">No board yet</span>'}
    </div>"""

    hole_section = f"""
    <div class="cards-row">
      <div class="label">Your hand</div>
      {hand_html(gs.our_hole)}
    </div>"""

    street = STREET_NAMES.get(gs.street, str(gs.street)).capitalize()
    info = f"""
    <div class="street-badge">{street}</div>
    <div class="pot-info">Pot: <strong>{gs.pot}</strong> chips &nbsp;|&nbsp; Stack: <strong>{gs.our_stack}</strong></div>"""

    if sess["hand_over"]:
        result = sess["result"] or {}
        winner = result.get("winner", "")
        winnings = result.get("winnings", 0)
        bot_hole = result.get("bot_hole", [])
        reason = result.get("reason", "")

        if winner == "human":
            result_html = f'<div class="result-win">You win <strong>{winnings}</strong> chips ({reason})</div>'
        elif winner == "bot":
            result_html = f'<div class="result-lose">Bot wins <strong>{winnings}</strong> chips ({reason})</div>'
        else:
            result_html = '<div class="result-tie">Tie</div>'

        bot_reveal = ""
        if bot_hole:
            bot_reveal = f"""
            <div class="cards-row">
              <div class="label">Bot's hand</div>
              {hand_html(bot_hole)}
            </div>"""

        return chips_bar + board_section + hole_section + info + result_html + bot_reveal + """
        <form hx-post="/deal" hx-target="#game-area" hx-swap="innerHTML" style="margin-top:12px">
          <button type="submit" class="btn btn-primary">Next Hand</button>
        </form>"""

    # Action buttons
    action_btns = ""
    for action in gs.valid_actions:
        cls = "btn-danger" if action == "FOLD" else "btn-success" if action.startswith("RAISE") else "btn-primary"
        label = action.replace("_", " ").title()
        action_btns += f"""
        <form hx-post="/action" hx-target="#game-area" hx-swap="innerHTML" hx-indicator="#thinking" style="display:inline">
          <input type="hidden" name="action" value="{action}">
          <button type="submit" class="btn {cls}">{label}</button>
        </form>"""

    log_html = ""
    if sess["log"]:
        entries = "\n".join(sess["log"][-10:])
        log_html = f'<div class="log">{entries}</div>'

    return chips_bar + board_section + hole_section + info + f"""
    <div class="action-btns">{action_btns}</div>
    <div id="thinking" class="htmx-indicator thinking"><span class="spinner"></span>Bot is thinking...</div>
    {log_html}"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sess = get_session(request)
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid or sid not in _sessions:
        sid, _ = new_session()
    response = templates.TemplateResponse(
        "index.html", {"request": request, "all_cards": ALL_CARDS}
    )
    set_session_cookie(response, sid)
    return response


@app.get("/game", response_class=HTMLResponse)
async def game_panel(request: Request, response: Response):
    sess = get_session(request)
    sid = request.cookies.get(SESSION_COOKIE)
    if not sess:
        sid, sess = new_session()
    html = render_game_panel(sess)
    resp = HTMLResponse(html)
    set_session_cookie(resp, sid)
    return resp


@app.post("/deal", response_class=HTMLResponse)
async def deal(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    sess = get_session(request)
    if not sess:
        sid, sess = new_session()

    pos = sess["human_position"]
    game: NLHEGame = sess["game"]
    gs = game.new_hand(human_position=pos)

    bot = Bot(budget_seconds=6.0)
    bot.new_hand(hole_cards=gs.our_hole[:], position=1 - pos)

    sess.update(
        bot=bot,
        state=gs,
        hand_started=True,
        hand_over=False,
        result=None,
        log=[f"New hand. You are {'SB' if pos == 0 else 'BB'}."],
        human_position=1 - pos,
    )

    html = render_game_panel(sess)
    resp = HTMLResponse(html)
    set_session_cookie(resp, sid)
    return resp


@app.post("/action", response_class=HTMLResponse)
async def action(request: Request, action: str = Form(...)):
    sid = request.cookies.get(SESSION_COOKIE)
    sess = get_session(request)
    if not sess or not sess.get("hand_started"):
        resp = HTMLResponse('<p style="color:#e74c3c">Session expired. <a href="/">Reload</a></p>')
        return resp

    game: NLHEGame = sess["game"]
    bot: Bot = sess["bot"]
    gs: GameState = sess["state"]

    sess["log"].append(f"You: {action}")
    result = game.human_action(action)

    if result["hand_over"]:
        winner = result.get("winner", "")
        winnings = result.get("winnings", 0)
        if winner == "human":
            sess["human_chips"] += winnings // 2
            sess["bot_chips"] -= winnings // 2
        elif winner == "bot":
            sess["human_chips"] -= winnings // 2
            sess["bot_chips"] += winnings // 2
        sess.update(hand_over=True, result=result)
    else:
        new_state: GameState = result["state"]
        sess["state"] = new_state
        bot.observe_action(action)
        bot_action = bot_decide(bot, new_state)
        sess["log"].append(f"Bot: {bot_action}")
        bot_result = game.bot_action(bot_action)

        if bot_result["hand_over"]:
            winner = bot_result.get("winner", "")
            winnings = bot_result.get("winnings", 0)
            if winner == "human":
                sess["human_chips"] += winnings // 2
                sess["bot_chips"] -= winnings // 2
            elif winner == "bot":
                sess["human_chips"] -= winnings // 2
                sess["bot_chips"] += winnings // 2
            sess.update(hand_over=True, result=bot_result)
        else:
            sess["state"] = bot_result.get("state", new_state)

    html = render_game_panel(sess)
    resp = HTMLResponse(html)
    set_session_cookie(resp, sid)
    return resp


@app.post("/solve", response_class=HTMLResponse)
async def solve(
    request: Request,
    hole1: str = Form(...),
    hole2: str = Form(...),
    position: int = Form(0),
    pot: int = Form(20),
    stack: int = Form(80),
    board: list[str] = Form(default=[]),
):
    our_hole = [hole1, hole2]
    board = [c for c in board if c not in our_hole][:5]

    # Validate: no duplicate cards
    all_selected = our_hole + board
    if len(set(all_selected)) != len(all_selected):
        return HTMLResponse('<p style="color:#e74c3c">Duplicate cards selected.</p>')

    if not board:
        # Preflop: read from table
        try:
            from nlhe.cfr.preflop import PreflopTable
            table = PreflopTable()
            idxs = tuple(sorted(card_str_to_idx(c) for c in our_hole))
            probs = table.lookup(idxs, position)
            action_names = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE", "RAISE_ALLIN"]
            recommended = action_names[int(np.argmax(probs))]
            return HTMLResponse(_render_bars(action_names, probs, recommended,
                                             title=f"Preflop: {hole1} {hole2} (CFR+ table)"))
        except Exception as e:
            return HTMLResponse(f'<p style="color:#e74c3c">Error: {e}</p>')

    # Postflop: run CFR solver
    from nlhe.cfr.solver import NLHESolver
    from nlhe.game import GameState, STREET_FLOP, STREET_TURN, STREET_RIVER
    from nlhe.cfr.equity import equity_mc

    street_map = {3: STREET_FLOP, 4: STREET_TURN, 5: STREET_RIVER}
    street = street_map.get(len(board), STREET_FLOP)
    valid_actions = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE", "RAISE_ALLIN"]

    state = GameState(
        street=street, pot=pot, our_stack=stack, opp_stack=stack,
        our_hole=our_hole, opp_hole=[], board=board,
        valid_actions=valid_actions, position=position, street_bets=[],
    )

    solver = NLHESolver(state, our_hole=our_hole, budget_seconds=8.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(solver.solve)
        try:
            recommended = future.result(timeout=9.0)
        except concurrent.futures.TimeoutError:
            recommended = "CALL"

    probs = solver._avg_strategy if solver._avg_strategy is not None \
            else np.ones(len(valid_actions)) / len(valid_actions)

    # EV estimates
    eq = equity_mc(our_hole, board, n_samples=500)
    ev_data = {}
    for a in valid_actions:
        if a == "FOLD":
            ev_data[a] = 0.0
        else:
            ev_data[a] = round((eq - 0.5) * pot, 1)

    board_str = " ".join(hand_html(board)) if board else ""
    title = f"{hole1} {hole2} | board: {' '.join(board)}"
    bars_html = _render_bars(valid_actions, probs, recommended, title=title)

    ev_rows = "".join(
        f'<tr class="{"recommended" if a == recommended else ""}"><td>{a}</td><td>{p:.1%}</td><td>{ev_data[a]:+.1f}</td></tr>'
        for a, p in zip(valid_actions, probs)
    )
    ev_table = f"""
    <table class="ev-table">
      <thead><tr><th>Action</th><th>Strategy</th><th>Est. EV (chips)</th></tr></thead>
      <tbody>{ev_rows}</tbody>
    </table>"""

    return HTMLResponse(bars_html + ev_table)


def _render_bars(actions: list[str], probs, recommended: str, title: str = "") -> str:
    title_html = f'<div style="font-weight:600;margin-bottom:8px">{title}</div>' if title else ""
    bars = ""
    for a, p in zip(actions, probs):
        pct = int(round(float(p) * 100))
        color_class = "fold" if a == "FOLD" else "call" if a == "CALL" else "raise"
        highlight = ' style="outline:2px solid #f39c12;outline-offset:2px"' if a == recommended else ""
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{a.replace('_', ' ')}</div>
          <div class="bar-track"{highlight}>
            <div class="bar-fill {color_class}" style="width:{max(pct,2)}%">
              <span class="bar-pct">{pct}%</span>
            </div>
          </div>
        </div>"""
    rec = f'<div style="margin-top:8px;font-size:0.85rem">CFR recommends: <strong>{recommended}</strong></div>'
    return title_html + f'<div class="bar-chart">{bars}</div>' + rec
