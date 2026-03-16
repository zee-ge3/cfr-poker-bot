# tests/test_opponent_profiler.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from io import StringIO


SAMPLE_CSV = """# Team 0: opponent_bot, Team 1: geoz
hand_number,street,active_team,team_0_bankroll,team_1_bankroll,action_type,action_amount,action_keep_1,action_keep_2,team_0_cards,team_1_cards,board_cards,team_0_discarded,team_1_discarded,team_0_bet,team_1_bet
0,Pre-Flop,0,0,0,RAISE,4,0,0,[],[],[],[],[],1,2
0,Pre-Flop,1,0,0,CALL,0,0,0,[],[],[],[],[],2,2
0,Flop,1,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2
0,Flop,0,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2
0,Flop,1,0,0,CHECK,0,0,0,[],[],[],[],[],2,2
0,Flop,0,0,0,RAISE,6,0,0,[],[],[],[],[],2,2
0,Flop,1,0,0,FOLD,0,0,0,[],[],[],[],[],2,2
1,Pre-Flop,1,6,-6,RAISE,4,0,0,[],[],[],[],[],1,2
1,Pre-Flop,0,6,-6,CALL,0,0,0,[],[],[],[],[],2,2
1,Flop,0,6,-6,DISCARD,0,0,1,[],[],[],[],[],2,2
1,Flop,1,6,-6,DISCARD,0,0,1,[],[],[],[],[],2,2
1,Flop,0,6,-6,RAISE,4,0,0,[],[],[],[],[],2,2
1,Flop,1,6,-6,CALL,0,0,0,[],[],[],[],[],2,2
"""


def _parse_csv(content):
    from opponent_profiler import parse_match_csv
    return parse_match_csv(content)


def test_geoz_slot_detection():
    """Correctly identifies geoz as team 1."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['geoz_slot'] == 1
    assert result['opp_name'] == 'opponent_bot'


def test_position_derivation_hand0():
    """Hand 0: first Pre-Flop actor is team 0 → team 0 is SB (IP post-flop)."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['hands'][0]['opp_is_ip'] == True  # opp=team0=SB=IP


def test_position_derivation_hand1():
    """Hand 1: first Pre-Flop actor is team 1 (geoz) → opp (team 0) is BB = OOP."""
    result = _parse_csv(SAMPLE_CSV)
    assert result['hands'][1]['opp_is_ip'] == False  # opp=team0=BB=OOP


def test_fold_to_raise_detected():
    """When geoz raises, opp fold events are detected correctly.
    - Hand 2 Flop: geoz raises, opp folds → folded=True
    - Hand 3 Flop: geoz raises, opp calls → folded=False
    """
    csv_with_fold = SAMPLE_CSV + (
        "2,Pre-Flop,0,0,0,RAISE,4,0,0,[],[],[],[],[],1,2\n"
        "2,Pre-Flop,1,0,0,CALL,0,0,0,[],[],[],[],[],2,2\n"
        "2,Flop,1,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "2,Flop,0,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "2,Flop,1,0,0,RAISE,6,0,0,[],[],[],[],[],2,2\n"
        "2,Flop,0,0,0,FOLD,0,0,0,[],[],[],[],[],2,2\n"
        "3,Pre-Flop,0,0,0,RAISE,4,0,0,[],[],[],[],[],1,2\n"
        "3,Pre-Flop,1,0,0,CALL,0,0,0,[],[],[],[],[],2,2\n"
        "3,Flop,1,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "3,Flop,0,0,0,DISCARD,0,0,1,[],[],[],[],[],2,2\n"
        "3,Flop,1,0,0,RAISE,6,0,0,[],[],[],[],[],2,2\n"
        "3,Flop,0,0,0,CALL,0,0,0,[],[],[],[],[],2,2\n"
    )
    result = _parse_csv(csv_with_fold)
    ftr_events = result['opp_ftr_events']
    # Hand 2 Flop: geoz raised, opp folded → folded=True
    # Hand 3 Flop: geoz raised, opp called → folded=False
    assert any(e['folded'] for e in ftr_events), \
        "Expected at least one fold-to-raise event"
    assert any(not e['folded'] for e in ftr_events), \
        "Expected at least one non-fold response to raise"
    # Both should be on Flop street
    assert all(e['street'] == 'Flop' for e in ftr_events)
