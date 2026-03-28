import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.features import load_dataset, get_train_test_split, FEATURE_COLS

os.makedirs("figures", exist_ok=True)

# ── Load data & models ────────────────────────────────────────────────────────
df = load_dataset()
X_train, X_test, y_train, y_test, _, _ = get_train_test_split(df)

# Get predictions from saved models
lr_model     = joblib.load("models_saved/logistic.pkl")
lr_scaler    = joblib.load("models_saved/logistic_scaler.pkl")
xgb_model    = joblib.load("models_saved/xgboost.pkl")
xgb_scaler   = joblib.load("models_saved/xgboost_scaler.pkl")

lr_pred      = lr_model.predict(lr_scaler.transform(X_test))
lr_prob      = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
xgb_pred     = xgb_model.predict(xgb_scaler.transform(X_test))
xgb_prob     = xgb_model.predict_proba(xgb_scaler.transform(X_test))[:, 1]

# LSTM predictions
from tensorflow.keras.models import load_model as keras_load
lstm_model   = keras_load("models_saved/lstm.keras")
lstm_scaler  = joblib.load("models_saved/lstm_scaler.pkl")
TIME_STEPS   = 5
X_all        = np.vstack([X_train, X_test])
X_all_scaled = lstm_scaler.transform(X_all)
X_test_scaled = X_all_scaled[len(X_train):]
X_seq = np.array([X_test_scaled[i:i+TIME_STEPS] for i in range(len(X_test_scaled)-TIME_STEPS)])
lstm_prob_seq = lstm_model.predict(X_seq, verbose=0).flatten()
lstm_pred_seq = (lstm_prob_seq > 0.5).astype(int)
y_test_lstm   = y_test[-len(lstm_pred_seq):]

STYLE = {
    "lr":   {"color": "#6366f1", "label": "Logistic Regression"},
    "xgb":  {"color": "#f59e0b", "label": "XGBoost"},
    "lstm": {"color": "#22c55e", "label": "LSTM"},
}
NAIVE = df["Target"].mean()

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0e1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Figure 1: Accuracy vs Balanced Accuracy vs Naive ─────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
models      = ["Logistic\nRegression", "XGBoost", "LSTM"]
raw_accs    = [
    (lr_pred == y_test).mean(),
    (xgb_pred == y_test).mean(),
    (lstm_pred_seq == y_test_lstm).mean(),
]
bal_accs    = [
    balanced_accuracy_score(y_test, lr_pred),
    balanced_accuracy_score(y_test, xgb_pred),
    balanced_accuracy_score(y_test_lstm, lstm_pred_seq),
]
x = np.arange(len(models))
w = 0.3
ax1.bar(x - w/2, [a*100 for a in raw_accs], w, label="Raw Accuracy",      color=["#6366f1","#f59e0b","#22c55e"], alpha=0.9)
ax1.bar(x + w/2, [a*100 for a in bal_accs], w, label="Balanced Accuracy", color=["#6366f1","#f59e0b","#22c55e"], alpha=0.5)
ax1.axhline(NAIVE*100, color="red",   linestyle="--", linewidth=1.5, label=f"Naive Baseline ({NAIVE*100:.1f}%)")
ax1.axhline(50,        color="white", linestyle=":",  linewidth=1,   label="Random (50%)")
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(40, 80)
ax1.set_title("Raw vs Balanced Accuracy", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9); ax1.set_facecolor("#1e2130")
for i, (r, b) in enumerate(zip(raw_accs, bal_accs)):
    ax1.text(i - w/2, r*100 + 0.5, f"{r*100:.1f}%", ha="center", fontsize=9)
    ax1.text(i + w/2, b*100 + 0.5, f"{b*100:.1f}%", ha="center", fontsize=9)

# ── Figure 2: Class Balance Pie ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
counts = df["Target"].value_counts()
ax2.pie(counts, labels=["Gap Up (1)", "Gap Down (0)"],
        colors=["#22c55e", "#ef4444"], autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10})
ax2.set_title("Dataset Class Balance", fontsize=13, fontweight="bold")

# ── Figure 3: Confusion Matrices ──────────────────────────────────────────────
for idx, (name, y_true, y_pred, color) in enumerate([
    ("Logistic Regression", y_test,       lr_pred,       "#6366f1"),
    ("XGBoost",             y_test,       xgb_pred,      "#f59e0b"),
    ("LSTM",                y_test_lstm,  lstm_pred_seq, "#22c55e"),
]):
    ax = fig.add_subplot(gs[1, idx])
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Gap Down", "Gap Up"]); ax.set_yticklabels(["Gap Down", "Gap Up"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    ax.set_facecolor("#1e2130")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")

# ── Figure 4: ROC Curves ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :2])
for key, y_true, prob in [
    ("lr",   y_test,      lr_prob),
    ("xgb",  y_test,      xgb_prob),
    ("lstm", y_test_lstm, lstm_prob_seq),
]:
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc     = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=STYLE[key]["color"], linewidth=2,
             label=f"{STYLE[key]['label']} (AUC = {roc_auc:.3f})")
ax4.plot([0,1],[0,1], "w--", linewidth=1, label="Random (AUC = 0.500)")
ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curves", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10); ax4.set_facecolor("#1e2130")

# ── Figure 5: XGBoost Feature Importance ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
fi   = xgb_model.feature_importances_
fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": fi}).sort_values("Importance")
colors_fi = ["#f59e0b"] * len(fi_df)
ax5.barh(fi_df["Feature"], fi_df["Importance"], color=colors_fi, alpha=0.9)
ax5.set_title("XGBoost Feature Importance", fontsize=11, fontweight="bold")
ax5.set_xlabel("Importance Score"); ax5.set_facecolor("#1e2130")

plt.suptitle("NIFTY Gap Prediction — Model Comparison", fontsize=16, fontweight="bold", y=1.01)
plt.savefig("figures/model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
plt.show()
print("Saved: figures/model_comparison.png")
