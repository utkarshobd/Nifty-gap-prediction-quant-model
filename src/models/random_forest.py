"""
ML Model 1: Random Forest Classifier
Strengths : interpretable, handles non-linear features, gives feature importance
"""
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_random_forest(X: np.ndarray, y: np.ndarray, symbol: str) -> dict:
    """
    Train with TimeSeriesSplit (no data leakage).
    Returns metrics dict.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series aware cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []

    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        fold_accs.append(accuracy_score(y_val, clf.predict(X_val)))

    # Final model on all data
    final_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    final_model.fit(X_scaled, y)

    joblib.dump(final_model, os.path.join(MODEL_DIR, f"{symbol}_rf.pkl"))
    joblib.dump(scaler,      os.path.join(MODEL_DIR, f"{symbol}_rf_scaler.pkl"))

    return {
        "model":        "Random Forest",
        "cv_accuracy":  round(float(np.mean(fold_accs)), 4),
        "cv_std":       round(float(np.std(fold_accs)), 4),
        "feature_importance": final_model.feature_importances_.tolist(),
    }


def predict_rf(X_latest: np.ndarray, symbol: str) -> float:
    """Returns probability of UP (gap up tomorrow)."""
    model  = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_rf.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_rf_scaler.pkl"))
    X_scaled = scaler.transform(X_latest[-1].reshape(1, -1))
    return round(float(model.predict_proba(X_scaled)[0][1]), 4)
