import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.features  import load_dataset, get_train_test_split, FEATURE_COLS
from src.models.logistic_model import train_logistic
from src.models.xgboost_model  import train_xgboost
from src.models.lstm_model     import train_lstm

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = load_dataset()
print(f"Rows: {len(df)} | Features: {FEATURE_COLS}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, train_df, test_df = get_train_test_split(df)
print(f"\nTrain: {len(X_train)} rows ({train_df['Date'].iloc[0].date()} to {train_df['Date'].iloc[-1].date()})")
print(f"Test:  {len(X_test)} rows ({test_df['Date'].iloc[0].date()} to {test_df['Date'].iloc[-1].date()})")

# ── Model 1: Logistic Regression ──────────────────────────────────────────────
print("\n[1/3] Training Logistic Regression...")
lr = train_logistic(X_train, y_train, X_test, y_test)
print(f"      Accuracy: {lr['accuracy']*100:.2f}%")
print(f"      Confusion Matrix: {lr['confusion_matrix']}")

# ── Model 2: XGBoost ──────────────────────────────────────────────────────────
print("\n[2/3] Training XGBoost...")
xgb = train_xgboost(X_train, y_train, X_test, y_test)
print(f"      Accuracy: {xgb['accuracy']*100:.2f}%")
print(f"      Confusion Matrix: {xgb['confusion_matrix']}")

# ── Model 3: LSTM ─────────────────────────────────────────────────────────────
print("\n[3/3] Training LSTM...")
lstm = train_lstm(X_train, y_train, X_test, y_test)
print(f"      Accuracy: {lstm['accuracy']*100:.2f}%")
print(f"      Confusion Matrix: {lstm['confusion_matrix']}")

# ── Summary ───────────────────────────────────────────────────────────────────
from sklearn.metrics import balanced_accuracy_score, f1_score
import pandas as pd
from src.utils.features import load_dataset, get_train_test_split, FEATURE_COLS

df = load_dataset()
_, _, _, y_test_raw, _, _ = get_train_test_split(df)

print("\n========== RESULTS ==========")
print(f"{'Model':<25} {'Accuracy':>10} {'Bal.Accuracy':>14} {'F1':>8}")
print("-" * 60)

for name, m, y_pred in [
    ("Logistic Regression", lr,   lr["y_pred"]),
    ("XGBoost",             xgb,  xgb["y_pred"]),
    ("LSTM",                lstm, lstm["y_pred"]),
]:
    # LSTM y_test is shorter by TIME_STEPS
    y_true = y_test_raw[-len(y_pred):]
    bal    = balanced_accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    print(f"{name:<25} {m['accuracy']*100:>9.2f}% {bal*100:>13.2f}% {f1:>8.4f}")

print()
print("Class balance in dataset:")
print(df["Target"].value_counts(normalize=True).round(3) * 100)
print("\nNaive baseline (always predict Gap Up):", round(df['Target'].mean()*100, 2), "%")
print("Models saved to models_saved/")
