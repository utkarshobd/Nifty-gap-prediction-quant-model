"""
ML Model 2: XGBoost Classifier
Strengths : gradient boosting, handles imbalanced data, fast, industry standard
"""
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_xgboost(X: np.ndarray, y: np.ndarray, symbol: str) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []

    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        fold_accs.append(accuracy_score(y_val, clf.predict(X_val)))

    # Final model
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    final_model.fit(X_scaled, y)

    joblib.dump(final_model, os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl"))
    joblib.dump(scaler,      os.path.join(MODEL_DIR, f"{symbol}_xgb_scaler.pkl"))

    return {
        "model":       "XGBoost",
        "cv_accuracy": round(float(np.mean(fold_accs)), 4),
        "cv_std":      round(float(np.std(fold_accs)), 4),
        "feature_importance": final_model.feature_importances_.tolist(),
    }


def predict_xgb(X_latest: np.ndarray, symbol: str) -> float:
    model  = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_xgb_scaler.pkl"))
    X_scaled = scaler.transform(X_latest[-1].reshape(1, -1))
    return round(float(model.predict_proba(X_scaled)[0][1]), 4)
