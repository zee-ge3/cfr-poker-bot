"""
SQLite diagnostic logger for GTOPoison laboratory analysis.
"""
import sqlite3
import os
from pathlib import Path

class SQLMatchLogger:
    def __init__(self, db_path="agent_logs/match_diagnostics.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # Add bot_version column for cross-version analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hands (
                match_id TEXT,
                hand_number INTEGER,
                bot_version TEXT,
                pnl FLOAT,
                cum_reward FLOAT,
                our_cards TEXT,
                board TEXT,
                opp_cards TEXT,
                opp_type TEXT,
                street INTEGER,
                is_showdown BOOLEAN,
                lockout BOOLEAN,
                opp_name TEXT,
                budget_used FLOAT,
                cfr_iters INTEGER,
                is_ip BOOLEAN
            )
        """)
        self.conn.commit()

    def log_hand(self, match_id, hand_num, bot_version, pnl, cum, our_c, board, opp_c, opp_t, street, is_sd, lockout,
                 opp_name=None, budget_used=None, cfr_iters=None, is_ip=None):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO hands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(match_id), hand_num, str(bot_version), pnl, cum, str(our_c), str(board), str(opp_c), opp_t, street, is_sd, lockout,
                  opp_name, budget_used, cfr_iters, is_ip))
            self.conn.commit()
        except Exception:
            pass

    def close(self):
        self.conn.close()
