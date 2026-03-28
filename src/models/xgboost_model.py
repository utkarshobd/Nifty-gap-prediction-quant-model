import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_xgboost(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    joblib.dump(model,  os.path.join(MODEL_DIR, "xgboost.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "xgboost_scaler.pkl"))

    return {
        "model":              "XGBoost",
        "accuracy":           round(float(acc), 4),
        "confusion_matrix":   cm.tolist(),
        "report":             classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": model.feature_importances_.tolist(),
        "y_pred":             y_pred,
    }


def predict_xgboost(X):
    model  = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "xgboost_scaler.pkl"))
    X_scaled = scaler.transform(X[-1].reshape(1, -1))
    return round(float(model.predict_proba(X_scaled)[0][1]), 4)
