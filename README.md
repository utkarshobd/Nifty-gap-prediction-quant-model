# Prediction Point 📈
> Stock Gap-Up / Gap-Down prediction using FinBERT + Random Forest + XGBoost + LSTM

## Tech Stack
| Layer | Tool |
|---|---|
| UI | Streamlit |
| NLP | FinBERT (ProsusAI/finbert) |
| ML Model 1 | Random Forest |
| ML Model 2 | XGBoost |
| DL Model | LSTM (TensorFlow/Keras) |
| Ensemble | Weighted Average (RF 25% + XGB 35% + LSTM 40%) |
| Stock Data | yfinance |
| News Data | Finnhub API (free tier) |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Finnhub API key
- Go to https://finnhub.io → Sign up → Copy your API key
- Edit `.env`:
```
FINNHUB_API_KEY=your_actual_key_here
```

### 3. Run the app
```bash
streamlit run app.py
```

## How to Use
1. Enter a stock symbol in the sidebar (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)
2. Set training history and news lookback days
3. Click **🚀 Run Full Pipeline**
4. View prediction, model comparison, sentiment analysis, and price charts

## Supported Symbols
- **US stocks**: `AAPL`, `TSLA`, `GOOGL`, `MSFT`, etc.
- **Indian stocks (NSE)**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `SBIN.NS`

## Project Structure
```
prediction-point/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt
├── .env                    # API keys (never commit this)
├── .streamlit/
│   └── config.toml         # Dark theme config
├── models_saved/           # Trained model files (auto-created)
└── src/
    ├── data/
    │   ├── stock_fetcher.py    # yfinance price data
    │   └── news_fetcher.py     # Finnhub news API
    ├── nlp/
    │   └── sentiment.py        # FinBERT scoring
    ├── models/
    │   ├── random_forest.py    # ML Model 1
    │   ├── xgboost_model.py    # ML Model 2
    │   ├── lstm_model.py       # DL Model
    │   └── ensemble.py         # Combines all 3
    └── utils/
        └── features.py         # Feature engineering
```

## Notes
- First run downloads FinBERT model (~400MB) — subsequent runs are fast
- Models are saved to `models_saved/` after training — no retraining needed unless you change the symbol
- TimeSeriesSplit is used for cross-validation to prevent data leakage
