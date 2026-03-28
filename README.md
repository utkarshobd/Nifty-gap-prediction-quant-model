# Prediction Point

> Multi-Factor NIFTY Gap Prediction using ML and Deep Learning

Predicts whether **NIFTY 50 will gap up or gap down** at next market open using global index signals.

## Pipeline

```
Dataset (CSV) → Time-based Split → Train 3 Models → Compare → Ensemble Prediction
```

### Step 1 — Dataset
Pre-built dataset: `data/nifty_gap_dataset_v2.csv`
- NIFTY 50 daily data from 2015 to 2026
- 2,394 rows after cleaning

| Feature | What it represents | Shift |
|---|---|---|
| SP500 % | Global sentiment | shift(1) |
| Nasdaq % | Tech / risk appetite | shift(1) |
| Nikkei % | Asia sentiment | shift(1) |
| HangSeng % | Asia sentiment | shift(1) |
| NIFTY Prev % | Local momentum | — |
| Volatility | 5-day rolling std of NIFTY returns | — |
| Target | 1 = Gap Up, 0 = Gap Down | — |

All external indices use `shift(1)` — dataset index is NIFTY dates, so previous trading day's close is used to avoid calendar mismatch and data leakage.

### Step 2 — Train-Test Split
Time-based 80/20 split (no shuffling, no leakage):
- Train: 2015 → ~2022
- Test:  ~2022 → 2026

### Step 3 — Models

| Model | Type | Why |
|---|---|---|
| Logistic Regression | Linear baseline | Interpretable, fast |
| XGBoost | Gradient Boosting | Best for tabular data, nonlinear patterns |
| LSTM | Deep Learning | Temporal sequence memory (5-day window) |

### Step 4 — Evaluation
- Test accuracy per model
- Confusion matrix
- Feature importance (XGBoost)

### Step 5 — Ensemble
```
P(UP) = 0.20 × LR + 0.50 × XGB + 0.30 × LSTM
```
XGBoost weighted highest — tree-based models outperform on structured tabular data.

### Expected Result
```
XGBoost > Logistic Regression > LSTM
```
Reason: Data is tabular and structured. Dataset size (~2400 rows) is too small for LSTM to learn meaningful temporal patterns.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| ML Model 1 | Logistic Regression (baseline) |
| ML Model 2 | XGBoost |
| DL Model | LSTM (TensorFlow/Keras) |
| Ensemble | Weighted Average (LR 20% + XGB 50% + LSTM 30%) |
| Data | yfinance (pre-downloaded) |

## Project Structure

```
prediction-point/
├── data/
│   └── nifty_gap_dataset_v2.csv   # pre-built dataset
├── models_saved/                  # trained models (auto-created, gitignored)
├── scripts/
│   └── generate_dataset.py        # one-off script to regenerate dataset
├── src/
│   ├── models/
│   │   ├── logistic_model.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   └── ensemble.py
│   └── utils/
│       └── features.py            # dataset loader + train/test split
├── .streamlit/
│   └── config.toml
├── app.py                         # Streamlit UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Resume Title

> Multi-Factor NIFTY Gap Prediction using ML and Deep Learning
