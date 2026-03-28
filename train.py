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
print("\n========== RESULTS ==========")
print(f"Logistic Regression : {lr['accuracy']*100:.2f}%")
print(f"XGBoost             : {xgb['accuracy']*100:.2f}%")
print(f"LSTM                : {lstm['accuracy']*100:.2f}%")
print("Models saved to models_saved/")
