"""
DL Model: LSTM (Long Short-Term Memory)
Strengths : remembers sequential patterns across 10 trading days,
            captures temporal dependencies that ML models miss
"""
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 10  # look back 10 trading days


def _build_lstm(n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _make_sequences(X: np.ndarray, y: np.ndarray):
    Xs, ys = [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i: i + SEQ_LEN])
        ys.append(y[i + SEQ_LEN])
    return np.array(Xs), np.array(ys)


def train_lstm(X: np.ndarray, y: np.ndarray, symbol: str) -> dict:
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = _make_sequences(X_scaled, y)

    # TimeSeriesSplit on sequences
    tscv = TimeSeriesSplit(n_splits=3)
    fold_accs = []

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
        y_tr, y_val = y_seq[train_idx], y_seq[val_idx]
        model = _build_lstm(X.shape[1])
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
            verbose=0,
        )
        preds = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        fold_accs.append(accuracy_score(y_val, preds))

    # Final model on all data
    final_model = _build_lstm(X.shape[1])
    final_model.fit(
        X_seq, y_seq,
        epochs=80,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    final_model.save(os.path.join(MODEL_DIR, f"{symbol}_lstm.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl"))

    return {
        "model":       "LSTM",
        "cv_accuracy": round(float(np.mean(fold_accs)), 4),
        "cv_std":      round(float(np.std(fold_accs)), 4),
    }


def predict_lstm(X_latest: np.ndarray, symbol: str) -> float:
    model_path  = os.path.join(MODEL_DIR, f"{symbol}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_scaler.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No LSTM model for {symbol}. Train first.")
    model  = load_model(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_latest)
    X_seq = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_latest.shape[1])
    return round(float(model.predict(X_seq, verbose=0)[0][0]), 4)
