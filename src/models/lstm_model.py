import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)

TIME_STEPS = 5


def _build_lstm(n_features):
    model = Sequential([
        LSTM(32, input_shape=(TIME_STEPS, n_features)),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def _make_sequences(X, y):
    Xs, ys = [], []
    for i in range(len(X) - TIME_STEPS):
        Xs.append(X[i:i + TIME_STEPS])
        ys.append(y[i + TIME_STEPS])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_all  = np.vstack([X_train, X_test])
    X_all_scaled = scaler.fit_transform(X_all)

    # Rebuild train/test from scaled full array preserving time order
    X_train_scaled = X_all_scaled[:len(X_train)]
    X_test_scaled  = X_all_scaled[len(X_train):]

    X_train_seq, y_train_seq = _make_sequences(X_train_scaled, y_train)
    X_test_seq,  y_test_seq  = _make_sequences(X_test_scaled,  y_test)

    model = _build_lstm(X_train.shape[1])
    model.fit(
        X_train_seq, y_train_seq,
        epochs=20,
        batch_size=16,
        verbose=0,
    )

    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    y_pred    = (model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
    cm        = confusion_matrix(y_test_seq, y_pred)

    model.save(os.path.join(MODEL_DIR, "lstm.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

    return {
        "model":            "LSTM",
        "accuracy":         round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "report":           classification_report(y_test_seq, y_pred, output_dict=True),
        "y_pred":           y_pred,
    }


def predict_lstm(X):
    model  = load_model(os.path.join(MODEL_DIR, "lstm.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
    X_scaled = scaler.transform(X)
    X_seq    = X_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, X.shape[1])
    return round(float(model.predict(X_seq, verbose=0)[0][0]), 4)
