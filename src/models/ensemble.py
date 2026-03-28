import numpy as np
from src.models.logistic_model import train_logistic, predict_logistic
from src.models.xgboost_model  import train_xgboost, predict_xgboost
from src.models.lstm_model     import train_lstm, predict_lstm
from src.utils.features        import FEATURE_COLS

# XGBoost weighted highest — best for tabular data
# Logistic is baseline, LSTM captures temporal patterns
WEIGHTS = {"lr": 0.20, "xgb": 0.50, "lstm": 0.30}


def train_all_models(X_train, y_train, X_test, y_test):
    lr_metrics  = train_logistic(X_train, y_train, X_test, y_test)
    xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    lstm_metrics = train_lstm(X_train, y_train, X_test, y_test)
    return {"lr": lr_metrics, "xgb": xgb_metrics, "lstm": lstm_metrics}


def predict_ensemble(X):
    lr_prob   = predict_logistic(X)
    xgb_prob  = predict_xgboost(X)
    lstm_prob = predict_lstm(X)

    ensemble_prob = (
        WEIGHTS["lr"]   * lr_prob +
        WEIGHTS["xgb"]  * xgb_prob +
        WEIGHTS["lstm"] * lstm_prob
    )
    ensemble_prob = round(ensemble_prob, 4)
    direction  = "GAP UP" if ensemble_prob >= 0.5 else "GAP DOWN"
    confidence = ensemble_prob if ensemble_prob >= 0.5 else round(1 - ensemble_prob, 4)

    return {
        "lr_prob":       lr_prob,
        "xgb_prob":      xgb_prob,
        "lstm_prob":     lstm_prob,
        "ensemble_prob": ensemble_prob,
        "direction":     direction,
        "confidence":    confidence,
    }
