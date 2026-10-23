import os
import joblib
import tensorflow as tf
from collections import deque
from command import Command
from buttons import Buttons
from device_utils import get_device
from match_logger import MatchLogger

# Verbose debug flag
DEBUG = os.environ.get('SF2_DEBUG', '0') == '1'

# Constants
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                  'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching',
                  'p1_in_move', 'p1_move_id', 'player2_id', 'p2_health', 'p2_x', 'p2_y',
                  'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                  'diff_x', 'diff_y', 'diff_health']

FEATURE_COLS = []
for t in range(WINDOW_SIZE-1, -1, -1):
    suffix = f"_t-{t}"
    for feat in STATE_FEATURES:
        FEATURE_COLS.append(feat + suffix)

BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']
ATTACK_BUTTONS = {'Y', 'B', 'X', 'A', 'L', 'R'}
DIRECTION_BUTTONS = {'UP', 'DOWN', 'RIGHT', 'LEFT'}
FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}

class Bot:
    def __init__(self, player_id=0, model_path=None):
        self.device = get_device()
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        self.loaded_character_id = None
        self.model = None
        self.scaler_mean_ = None
        self.scaler_scale_ = None
        self._compiled_predict = None
        
        # Guardrail state tracking
        self.recent_attacks = deque()  # stores (frame_idx, button) for 10s anti-spam
        self.idle_frames = 0           # counts consecutive frames with no action
        self.frame_counter = 0
        
        # Match logger (Rule 9 compliance)
        self.logger = None

        # Frame buffer (6 frames)
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self._init_buffer()

        # Load initial character model
        self.load_character(player_id, model_path)

    def _init_buffer(self):
        self.buffer.clear()
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())

    def load_character(self, character_id, custom_model_path=None):
        cid = int(character_id)
        if self.loaded_character_id == cid and self.model is not None:
            return

        model_path = custom_model_path or os.path.join(self.models_dir, f'model_{cid}.keras')
        scaler_path = model_path + '.scaler'

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"[Bot Warning] Model or scaler for character {cid} not found! Falling back to character 0.")
            cid = 0
            model_path = os.path.join(self.models_dir, 'model_0.keras')
            scaler_path = model_path + '.scaler'

        print(f"[Bot] Loading model & scaler for Character {cid} from {model_path}...")
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(model_path)

        scaler = joblib.load(scaler_path)
        self.scaler_mean_ = tf.constant(scaler.mean_, dtype=tf.float32)
        self.scaler_scale_ = tf.constant(scaler.scale_, dtype=tf.float32)

        n_features = len(FEATURE_COLS)

        def _make_compiled_predict():
            @tf.function(
                input_signature=[tf.TensorSpec(shape=[1, n_features], dtype=tf.float32)],
                jit_compile=False
            )
            def compiled_predict(x):
                x_scaled = (x - self.scaler_mean_) / self.scaler_scale_
                return self.model(x_scaled, training=False)
            return compiled_predict

        with tf.device(self.device):
            self._compiled_predict = _make_compiled_predict()
            dummy = tf.zeros((1, n_features), dtype=tf.float32)
            for _ in range(2):
                _ = self._compiled_predict(dummy)

        self.loaded_character_id = cid
        self._init_buffer()
        print(f"[Bot] Character {cid} model warmed up and ready.")

    def _frame_to_dict(self, gs, player_side):
        # Resolve perspective: self vs opponent
        is_p1 = (str(player_side) == "1")
        self_p = gs.player1 if is_p1 else gs.player2
        opp_p = gs.player2 if is_p1 else gs.player1

        # Coordinate differences from self perspective
        diff_x = self_p.x_coord - opp_p.x_coord
        diff_y = self_p.y_coord - opp_p.y_coord
        diff_health = self_p.health - opp_p.health

        d = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'has_round_started': int(gs.has_round_started),
            'is_round_over': int(gs.is_round_over),
            'player1_id': self_p.player_id,
            'p1_health': self_p.health,
            'p1_x': self_p.x_coord,
            'p1_y': self_p.y_coord,
            'p1_jumping': int(self_p.is_jumping),
            'p1_crouching': int(self_p.is_crouching),
            'p1_in_move': int(self_p.is_player_in_move),
            'p1_move_id': self_p.move_id,
            'player2_id': opp_p.player_id,
            'p2_health': opp_p.health,
            'p2_x': opp_p.x_coord,
            'p2_y': opp_p.y_coord,
            'p2_jumping': int(opp_p.is_jumping),
            'p2_crouching': int(opp_p.is_crouching),
            'p2_in_move': int(opp_p.is_player_in_move),
            'p2_move_id': opp_p.move_id,
            'diff_x': diff_x,
            'diff_y': diff_y,
            'diff_health': diff_health,
        }
        return d, diff_x

    def fight(self, gs, player_id):
        self.frame_counter += 1
        player_side = str(player_id)
        
        # Initialize logger on first call
        if self.logger is None:
            self.logger = MatchLogger(player_side=player_side)

        # 1. Identify our character and switch model dynamically if needed (Rule 3)
        our_char_id = gs.player1.player_id if player_side == "1" else gs.player2.player_id
        if self.loaded_character_id != our_char_id:
            self.load_character(our_char_id)

        # 2. Append new frame
        raw, diff_x = self._frame_to_dict(gs, player_side)
        self.buffer.append(raw)

        # 3. Build flattened features
        flat = []
        for t in range(WINDOW_SIZE-1, -1, -1):
            frame = self.buffer[WINDOW_SIZE-1 - t]
            for feat in STATE_FEATURES:
                val = frame[feat]
                flat.append(FIGHT_MAP[val] if feat == 'fight_result' else int(val))

        # 4. Neural Network Inference
        x = tf.constant([flat], dtype=tf.float32)
        preds = self._compiled_predict(x).numpy()[0]

        # 5. Probability mapping & Rule 7 (Opposite direction prevention)
        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        
        # Directional side adjustment:
        # If agent is to the right of opponent (diff_x > 0), advance is LEFT, retreat is RIGHT.
        # Since models were trained with P1 on left (diff_x < 0 advancing RIGHT), swap directions if diff_x > 0
        if diff_x > 0:
            probs['LEFT'], probs['RIGHT'] = probs['RIGHT'], probs['LEFT']

        if probs['LEFT'] and probs['RIGHT']:
            if probs['LEFT'] > probs['RIGHT']:
                probs['RIGHT'] = 0.0
            else:
                probs['LEFT'] = 0.0
        if probs['UP'] and probs['DOWN']:
            if probs['UP'] > probs['DOWN']:
                probs['DOWN'] = 0.0
            else:
                probs['UP'] = 0.0

        # Raw button activations threshold
        btn_map = {b: (probs[b] > 0.005) for b in BUTTONS}

        # 6. Rule 4: Anti-Spam Guardrail (<8 same attack in 10s window)
        # Prune attacks older than 600 frames (10 seconds at 60 FPS)
        cutoff = self.frame_counter - 600
        while self.recent_attacks and self.recent_attacks[0][0] < cutoff:
            self.recent_attacks.popleft()

        # Count frequencies of each attack button
        attack_counts = {}
        for _, btn in self.recent_attacks:
            attack_counts[btn] = attack_counts.get(btn, 0) + 1

        for btn in ATTACK_BUTTONS:
            if btn_map.get(btn, False):
                if attack_counts.get(btn, 0) >= 7:
                    # Suppress repeating attack to avoid Rule 4 disqualification
                    btn_map[btn] = False
                else:
                    self.recent_attacks.append((self.frame_counter, btn))
                    attack_counts[btn] = attack_counts.get(btn, 0) + 1

        # 7. Rule 5: Anti-Idle Watchdog (<3s consecutive inactivity)
        has_action = any(btn_map.values())
        self_p = gs.player1 if player_side == "1" else gs.player2
        is_disabled = self_p.is_jumping or self_p.is_player_in_move

        if not has_action and not is_disabled and gs.has_round_started and not gs.is_round_over:
            self.idle_frames += 1
            if self.idle_frames >= 120:  # ~2 seconds idle threshold
                # Inject active forward movement to prevent standing around penalty
                if diff_x < 0:
                    btn_map['RIGHT'] = True
                else:
                    btn_map['LEFT'] = True
                self.idle_frames = 0
        else:
            self.idle_frames = 0

        # 8. Rule 7: Strictly enforce NO Start and NO Select
        btn_map['START'] = False
        btn_map['SELECT'] = False

        # 9. Create Command object
        cmd = Command()
        if player_side == "1":
            cmd.player_buttons = Buttons(btn_map)
        else:
            cmd.player2_buttons = Buttons(btn_map)

        # 10. Rule 9: Match Logging
        active_list = [b for b, state in btn_map.items() if state]
        self.logger.log_frame(gs, active_list, our_char_id)

        if DEBUG:
            print(f"[Bot] Active buttons for Player {player_side}:", ", ".join(active_list) if active_list else "None")

        return cmd
