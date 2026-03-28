import pandas as pd
import numpy as np


FEATURE_COLS = [
    "sentiment",       # FinBERT daily score
    "gap_pct",         # today's gap %
    "price_change",    # close-to-close %
    "volume_change",   # volume % change
    "ma5",             # 5-day moving average
    "ma10",            # 10-day moving average
    "ma_ratio",        # ma5 / ma10 — trend signal
    "volatility",      # 5-day rolling std
    "rsi",             # Relative Strength Index (14-day)
]


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(price_df: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price data with daily sentiment and compute technical features.
    Target = next day's direction (shift -1).
    """
    df = price_df.copy()

    # Merge sentiment — fill missing dates with 0 (neutral)
    if not daily_sentiment.empty:
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(daily_sentiment, on="date", how="left")
        df["sentiment"] = df["sentiment"].fillna(0.0)
    else:
        df["sentiment"] = 0.0

    # Technical indicators
    df["price_change"]  = df["close"].pct_change() * 100
    df["volume_change"] = df["volume"].pct_change() * 100
    df["ma5"]           = df["close"].rolling(5).mean()
    df["ma10"]          = df["close"].rolling(10).mean()
    df["ma_ratio"]      = df["ma5"] / (df["ma10"] + 1e-9)
    df["volatility"]    = df["close"].rolling(5).std()
    df["rsi"]           = _compute_rsi(df["close"])

    # Target: NEXT day's direction
    df["target"] = df["direction"].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
