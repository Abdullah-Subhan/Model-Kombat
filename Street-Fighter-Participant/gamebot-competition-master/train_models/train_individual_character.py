import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Defining constants
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over',
                  'player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching',
                  'p1_in_move', 'p1_move_id', 'player2_id', 'p2_health', 'p2_x', 'p2_y',
                  'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                  'diff_x', 'diff_y', 'diff_health']

# Feature columns
FEATURE_COLS = []
for t in range(WINDOW_SIZE-1, -1, -1):
    suffix = f"_t-{t}"
    for feat in STATE_FEATURES:
        FEATURE_COLS.append(feat + suffix)

BUTTONS = ['up', 'down', 'right', 'left', 'y', 'b', 'x', 'a', 'l', 'r']
P1_BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS]

def build_residual_model(input_dim, output_dim):
    inputs = layers.Input(shape=(input_dim,))
    d1 = layers.Dense(256, activation='relu')(inputs)
    # Residual Block
    res1 = layers.Dense(256, activation='relu')(d1)
    res1 = layers.Dropout(0.2)(res1)
    res1 = layers.Dense(256, activation='relu')(res1)
    add1 = layers.Add()([d1, res1])
    # Transition
    d2 = layers.Dense(128, activation='relu')(add1)
    d2 = layers.Dropout(0.2)(d2)
    d3 = layers.Dense(64, activation='relu')(d2)
    outputs = layers.Dense(output_dim, activation='sigmoid')(d3)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def train_model(csv_path: str, model_path: str):
    df = pd.read_csv(csv_path)

    # Balanced sampling
    df_pos = df[df[P1_BUTTON_COLS].sum(axis=1) > 0]
    df_neg_all = df[df[P1_BUTTON_COLS].sum(axis=1) == 0]
    neg_count = min(len(df_neg_all), len(df_pos) * 2)
    df_neg = df_neg_all.sample(n=neg_count, random_state=42)
    df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}
    BOOL_MAP = {False: 0, True: 1, 'False': 0, 'True': 1}
    for t in range(WINDOW_SIZE):
        col_fr = f'fight_result_t-{t}'
        df[col_fr] = df[col_fr].map(FIGHT_MAP)
        for bf in ['has_round_started','is_round_over','p1_jumping','p1_crouching','p1_in_move','p2_jumping','p2_crouching','p2_in_move']:
            col_b = f'{bf}_t-{t}'
            df[col_b] = df[col_b].map(BOOL_MAP)

    X = df[FEATURE_COLS]
    y = df[P1_BUTTON_COLS].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, model_path + '.scaler')
    print(f"Saved scaler to {model_path}.scaler")

    sample_weight = (y_train.sum(axis=1) > 0).astype(float) * 3 + 1
    model = build_residual_model(len(FEATURE_COLS), len(P1_BUTTON_COLS))

    ckpt = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weight,
        validation_data=(X_val_scaled, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[ckpt, early_stop]
    )
    print(f"Training done. Best model saved at {model_path}")

if __name__ == '__main__':
    # Train all 12 characters by default, or specify desired character IDs
    characters_to_train = list(range(12))

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flattened_window_datasets'))
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(out, exist_ok=True)
    
    for cid in characters_to_train:
        fn = f'windowed_dataset_{cid}.csv'
        if not os.path.exists(os.path.join(base, fn)):
            print(f"\n=== Skipping character {cid} - dataset not found ===")
            continue
            
        inp = os.path.join(base, fn)
        mdl = os.path.join(out, f'model_{cid}.keras')
        print(f"\n=== Training character {cid} ===")
        train_model(inp, mdl)
