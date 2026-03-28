import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nifty_gap_dataset_v2.csv")

FEATURE_COLS = ["SP500 %", "Nasdaq %", "Nikkei %", "HangSeng %", "NIFTY Prev %", "Volatility"]
TARGET_COL   = "Target"


def load_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    X_train = train[FEATURE_COLS].values.astype("float32")
    y_train = train[TARGET_COL].values.astype("int32")
    X_test  = test[FEATURE_COLS].values.astype("float32")
    y_test  = test[TARGET_COL].values.astype("int32")

    return X_train, X_test, y_train, y_test, train, test


def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)
