"""
Ensemble: combines RF + XGBoost + LSTM predictions.

Why ensemble works better:
- RF    : captures non-linear feature interactions
- XGBoost: captures gradient-boosted patterns, handles noise
- LSTM  : captures temporal sequence memory (10-day window)
Each model sees the problem differently → averaging reduces individual errors.
"""
import numpy as np
from src.models.random_forest  import train_random_forest, predict_rf
from src.models.xgboost_model  import train_xgboost, predict_xgb
from src.models.lstm_model     import train_lstm, predict_lstm
from src.utils.features        import FEATURE_COLS


# Weights reflect each model's typical contribution
WEIGHTS = {"rf": 0.25, "xgb": 0.35, "lstm": 0.40}


def train_all_models(df, symbol: str) -> dict:
    """Train all 3 models and return their individual metrics."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(np.int32)

    if len(df) < 50:
        raise ValueError(f"Need at least 50 rows to train. Got {len(df)}.")

    rf_metrics   = train_random_forest(X, y, symbol)
    xgb_metrics  = train_xgboost(X, y, symbol)
    lstm_metrics = train_lstm(X, y, symbol)

    return {
        "rf":   rf_metrics,
        "xgb":  xgb_metrics,
        "lstm": lstm_metrics,
        "rows": len(df),
    }


def predict_ensemble(df, symbol: str) -> dict:
    """
    Run all 3 models and return:
    - individual probabilities
    - ensemble probability
    - final direction + confidence
    """
    X = df[FEATURE_COLS].values.astype(np.float32)

    rf_prob   = predict_rf(X, symbol)
    xgb_prob  = predict_xgb(X, symbol)
    lstm_prob = predict_lstm(X, symbol)

    ensemble_prob = (
        WEIGHTS["rf"]   * rf_prob +
        WEIGHTS["xgb"]  * xgb_prob +
        WEIGHTS["lstm"] * lstm_prob
    )
    ensemble_prob = round(ensemble_prob, 4)
    direction     = "GAP UP ▲" if ensemble_prob >= 0.5 else "GAP DOWN ▼"
    confidence    = ensemble_prob if ensemble_prob >= 0.5 else round(1 - ensemble_prob, 4)

    return {
        "rf_prob":       rf_prob,
        "xgb_prob":      xgb_prob,
        "lstm_prob":     lstm_prob,
        "ensemble_prob": ensemble_prob,
        "direction":     direction,
        "confidence":    confidence,
    }
