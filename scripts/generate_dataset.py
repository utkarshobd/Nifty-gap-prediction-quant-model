import pandas as pd
import yfinance as yf

# -------------------------
# 1. Parameters
# -------------------------
start, end = "2015-01-01", "2026-03-26"

# -------------------------
# 2. Download Data
# -------------------------
nifty    = yf.download("^NSEI",  start=start, end=end, progress=False, auto_adjust=True)
sp500    = yf.download("^GSPC",  start=start, end=end, progress=False, auto_adjust=True)
nasdaq   = yf.download("^IXIC",  start=start, end=end, progress=False, auto_adjust=True)
nikkei   = yf.download("^N225",  start=start, end=end, progress=False, auto_adjust=True)
hangseng = yf.download("^HSI",   start=start, end=end, progress=False, auto_adjust=True)

# -------------------------
# 3. Clean Columns
# -------------------------
for d in [nifty, sp500, nasdaq, nikkei, hangseng]:
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

nifty    = nifty[["Open", "Close"]].copy()
sp500    = sp500[["Close"]].rename(columns={"Close": "SP500"})
nasdaq   = nasdaq[["Close"]].rename(columns={"Close": "Nasdaq"})
nikkei   = nikkei[["Close"]].rename(columns={"Close": "Nikkei"})
hangseng = hangseng[["Close"]].rename(columns={"Close": "HangSeng"})

# -------------------------
# 4. Merge
# -------------------------
df = nifty.join(sp500, how="inner") \
          .join(nasdaq, how="inner") \
          .join(nikkei, how="inner") \
          .join(hangseng, how="inner")

df = df.sort_index()

# -------------------------
# 5. Feature Engineering
# -------------------------

# Gap %
df["Prev Close"] = df["Close"].shift(1)
df["Gap %"] = (df["Open"] - df["Prev Close"]) / df["Prev Close"] * 100

# -------------------------
# 🔥 CRITICAL: SHIFT ALL EXTERNAL MARKETS
# -------------------------

df["SP500 %"]    = df["SP500"].pct_change().shift(1) * 100
df["Nasdaq %"]   = df["Nasdaq"].pct_change().shift(1) * 100
df["Nikkei %"]   = df["Nikkei"].pct_change().shift(1) * 100
df["HangSeng %"] = df["HangSeng"].pct_change().shift(1) * 100

# Local features
df["NIFTY Prev %"] = df["Close"].pct_change() * 100
df["Volatility"]   = df["Close"].pct_change().rolling(5).std() * 100

# -------------------------
# 6. Clean Dataset
# -------------------------
df = df.dropna()

# Classification target: 1 = gap up, 0 = gap down
df["Target"] = (df["Gap %"] > 0).astype(int)

# Final columns
df = df[[
    "Gap %",
    "SP500 %",
    "Nasdaq %",
    "Nikkei %",
    "HangSeng %",
    "NIFTY Prev %",
    "Volatility",
    "Target"
]]

df.reset_index(inplace=True)

# -------------------------
# 7. Save
# -------------------------
df.to_csv("nifty_gap_dataset_v2.csv", index=False)

print("Dataset created: nifty_gap_dataset_v2.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())