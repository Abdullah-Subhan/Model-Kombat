import os
import csv
import time

class MatchLogger:
    """
    Automatic Match Logger adhering strictly to Rule 9 of Participant Guidelines.
    Saves CSV logs for every fight in the match computer at gamebot-competition-master/match_logs/
    Log columns: frame, time, player_side, character_id, buttons_pressed, p1_health, p2_health
    """
    def __init__(self, player_side="1"):
        self.player_side = str(player_side)
        self.p_tag = f"P{self.player_side}"
        self.logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'match_logs'))
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.current_fight_num = self._get_next_fight_number()
        self.current_file = None
        self.writer = None
        self.file_handle = None
        self.frame_count = 0
        self.in_fight = False

    def _get_next_fight_number(self):
        existing = os.listdir(self.logs_dir)
        count = 1
        for f in existing:
            if f.endswith(f"_{self.p_tag}.csv"):
                count += 1
        return count

    def start_new_fight(self):
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass
        
        filename = f"stage1_fight{self.current_fight_num}_{self.p_tag}.csv"
        self.current_file = os.path.join(self.logs_dir, filename)
        self.file_handle = open(self.current_file, mode='w', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(['frame', 'time', 'player_side', 'character_id', 'buttons_pressed', 'p1_health', 'p2_health'])
        self.file_handle.flush()
        print(f"[MatchLogger] Started match log: {filename}")
        self.frame_count = 0
        self.in_fight = True
        self.current_fight_num += 1

    def log_frame(self, gs, active_buttons, character_id):
        # Detect fight boundaries
        if gs.has_round_started and not self.in_fight:
            self.start_new_fight()
        elif gs.is_round_over and self.in_fight:
            # End of current fight round
            self.in_fight = False
            
        if not self.in_fight or not self.file_handle:
            return

        self.frame_count += 1
        btn_str = "+".join(active_buttons) if active_buttons else "NONE"
        p1_h = gs.player1.health
        p2_h = gs.player2.health
        
        self.writer.writerow([
            self.frame_count,
            gs.timer,
            self.p_tag,
            character_id,
            btn_str,
            p1_h,
            p2_h
        ])
        if self.frame_count % 30 == 0:
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None
