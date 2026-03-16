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


def test_ftr_aggregation():
    """FTR rate computed correctly from multiple hands."""
    from opponent_profiler import aggregate_opponent

    ftr_events = [
        {'street': 'Flop', 'opp_is_ip': False, 'folded': True},
        {'street': 'Flop', 'opp_is_ip': False, 'folded': True},
        {'street': 'Flop', 'opp_is_ip': False, 'folded': False},
        {'street': 'Turn', 'opp_is_ip': True,  'folded': False},
    ]
    match_data = {
        'opp_name': 'TestBot',
        'hands': [{'hand_num': i, 'opp_is_ip': False} for i in range(120)],
        'opp_ftr_events': ftr_events,
        'rows': [],
        'opp_slot': 0,
        'geoz_slot': 1,
    }
    profile = aggregate_opponent([match_data])
    # Flop OOP: 2 folds out of 3 raises = 66.7%
    assert abs(profile['ftr_oop_Flop'] - 2/3) < 0.01
    # Turn IP: 0 folds out of 1 = 0% BUT n=1 < 3, so should be nan
    import math
    assert math.isnan(profile['ftr_ip_Turn'])
    assert profile['total_hands'] == 120


def test_pf_fold_rate():
    """Pre-flop fold rate computed from action rows."""
    from opponent_profiler import aggregate_opponent
    result = _parse_csv(SAMPLE_CSV)
    profile = aggregate_opponent([result])
    # Hand 0: opp (team 0) raised pre-flop (not a fold)
    # Hand 1: opp (team 0) called pre-flop (not a fold)
    assert profile['pf_fold_rate'] == 0.0


def test_opponent_type_classification():
    from opponent_profiler import classify_opponent_type
    # Maniac: very low PF fold
    assert classify_opponent_type(pf_fold_rate=0.05, vpip=0.95) == 'maniac'
    # LAG: moderate-low PF fold, high VPIP
    assert classify_opponent_type(pf_fold_rate=0.25, vpip=0.75) == 'lag'
    # TAG: moderate PF fold
    assert classify_opponent_type(pf_fold_rate=0.45, vpip=0.55) == 'tag'
    # Calling station: high VPIP, low raise rate
    assert classify_opponent_type(pf_fold_rate=0.05, vpip=0.95, raise_rate=0.05) == 'calling_station'
