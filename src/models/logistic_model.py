import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_logistic(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    joblib.dump(model,  os.path.join(MODEL_DIR, "logistic.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "logistic_scaler.pkl"))

    return {
        "model":        "Logistic Regression",
        "accuracy":     round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "report":       classification_report(y_test, y_pred, output_dict=True),
        "y_pred":       y_pred,
    }


def predict_logistic(X):
    model  = joblib.load(os.path.join(MODEL_DIR, "logistic.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "logistic_scaler.pkl"))
    X_scaled = scaler.transform(X[-1].reshape(1, -1))
    return round(float(model.predict_proba(X_scaled)[0][1]), 4)
