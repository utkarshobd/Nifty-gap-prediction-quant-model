import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data.stock_fetcher  import fetch_stock_data
from src.data.news_fetcher   import fetch_news
from src.nlp.sentiment       import score_articles, get_daily_sentiment, load_finbert
from src.utils.features      import build_features, FEATURE_COLS
from src.models.ensemble     import train_all_models, predict_ensemble, WEIGHTS
from src.models.random_forest import predict_rf
from src.models.xgboost_model import predict_xgb
from src.models.lstm_model    import predict_lstm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Point",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130; border-radius: 12px;
        padding: 20px; text-align: center; border: 1px solid #2d3250;
    }
    .up   { color: #22c55e; font-size: 2rem; font-weight: 800; }
    .down { color: #ef4444; font-size: 2rem; font-weight: 800; }
    .model-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Prediction Point")
    st.caption("Stock Gap Prediction · ML + DL + NLP")
    st.divider()

    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Use .NS suffix for Indian stocks (e.g. RELIANCE.NS)",
    ).upper().strip()

    days = st.slider("Training History (days)", 200, 730, 365, 50)
    news_days = st.slider("News Lookback (days)", 7, 60, 30)

    st.divider()
    st.markdown("**Model Weights (Ensemble)**")
    st.caption(f"RF: {WEIGHTS['rf']} · XGB: {WEIGHTS['xgb']} · LSTM: {WEIGHTS['lstm']}")
    st.divider()

    run_pipeline = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
    st.caption("Fetches data → scores sentiment → trains all 3 models → predicts")

# ── Session state ──────────────────────────────────────────────────────────────
for key in ["trained", "metrics", "prediction", "feature_df", "articles_df", "price_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Main pipeline ──────────────────────────────────────────────────────────────
if run_pipeline:
    st.session_state.trained = False

    # Step 1: Fetch prices
    with st.status("📊 Fetching stock prices...", expanded=True) as status:
        try:
            price_df = fetch_stock_data(symbol, days=days)
            st.write(f"✅ {len(price_df)} trading days loaded for **{symbol}**")
            st.session_state.price_df = price_df
        except Exception as e:
            st.error(f"Price fetch failed: {e}")
            st.stop()

    # Step 2: Fetch & score news
    with st.status("📰 Fetching & scoring news with FinBERT...", expanded=True) as status:
        try:
            articles = fetch_news(symbol, days_back=news_days)
            if articles:
                st.write(f"✅ {len(articles)} articles fetched")
                with st.spinner("Loading FinBERT model (first run downloads ~400MB)..."):
                    articles_df = score_articles(articles)
                daily_sentiment = get_daily_sentiment(articles_df)
                st.write(f"✅ Sentiment scored across {len(daily_sentiment)} days")
                st.session_state.articles_df = articles_df
            else:
                st.warning("No news found — using neutral sentiment (0.0)")
                articles_df = pd.DataFrame()
                daily_sentiment = pd.DataFrame(columns=["date", "sentiment"])
                st.session_state.articles_df = articles_df
        except ValueError as e:
            st.warning(f"News skipped: {e}. Using neutral sentiment.")
            articles_df = pd.DataFrame()
            daily_sentiment = pd.DataFrame(columns=["date", "sentiment"])
            st.session_state.articles_df = articles_df

    # Step 3: Build features
    with st.status("⚙️ Engineering features...", expanded=True) as status:
        feature_df = build_features(price_df, daily_sentiment)
        st.write(f"✅ Feature matrix: {feature_df.shape[0]} rows × {len(FEATURE_COLS)} features")
        st.session_state.feature_df = feature_df

    # Step 4: Train all 3 models
    with st.status("🧠 Training RF + XGBoost + LSTM...", expanded=True) as status:
        try:
            metrics = train_all_models(feature_df, symbol)
            st.write(f"✅ Random Forest  — CV Accuracy: **{metrics['rf']['cv_accuracy']*100:.1f}%**")
            st.write(f"✅ XGBoost        — CV Accuracy: **{metrics['xgb']['cv_accuracy']*100:.1f}%**")
            st.write(f"✅ LSTM           — CV Accuracy: **{metrics['lstm']['cv_accuracy']*100:.1f}%**")
            st.session_state.metrics = metrics
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # Step 5: Predict
    with st.status("🔮 Generating ensemble prediction...", expanded=True) as status:
        prediction = predict_ensemble(feature_df, symbol)
        st.write(f"✅ Prediction: **{prediction['direction']}** with {prediction['confidence']*100:.1f}% confidence")
        st.session_state.prediction = prediction
        st.session_state.trained = True

# ── Dashboard ──────────────────────────────────────────────────────────────────
if st.session_state.trained:
    pred    = st.session_state.prediction
    metrics = st.session_state.metrics
    feat_df = st.session_state.feature_df
    art_df  = st.session_state.articles_df
    price_df = st.session_state.price_df

    st.divider()

    # ── Section 1: Final Prediction ──────────────────────────────────────────
    st.subheader("🔮 Tomorrow's Prediction")
    is_up = "UP" in pred["direction"]
    color = "#22c55e" if is_up else "#ef4444"
    arrow = "▲" if is_up else "▼"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af; font-size:0.9rem; margin-bottom:8px">
                {symbol} · Next Day Gap Prediction
            </div>
            <div style="color:{color}; font-size:3rem; font-weight:900; line-height:1">
                {arrow} {"GAP UP" if is_up else "GAP DOWN"}
            </div>
            <div style="color:#e5e7eb; font-size:1.1rem; margin-top:8px">
                Ensemble Confidence: <b>{pred['confidence']*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gauge chart for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred["ensemble_prob"] * 100,
            title={"text": "UP Probability", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar":  {"color": color},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0, 50],  "color": "#2d1b1b"},
                    {"range": [50, 100],"color": "#1b2d1b"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig_gauge.update_layout(
            height=220, paper_bgcolor="#0e1117", font_color="white",
            margin=dict(t=40, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown("**Individual Model Votes**")
        models_data = {
            "Random Forest": pred["rf_prob"],
            "XGBoost":       pred["xgb_prob"],
            "LSTM":          pred["lstm_prob"],
        }
        for name, prob in models_data.items():
            vote = "▲ UP" if prob >= 0.5 else "▼ DOWN"
            vote_color = "#22c55e" if prob >= 0.5 else "#ef4444"
            st.markdown(
                f"**{name}** — "
                f"<span style='color:{vote_color}'>{vote}</span> "
                f"({prob*100:.1f}%)",
                unsafe_allow_html=True,
            )
            st.progress(prob)

    st.divider()

    # ── Section 2: Model Comparison ──────────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")
    st.caption("TimeSeriesSplit cross-validation — no data leakage")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    model_list = [
        ("Random Forest", metrics["rf"],   "#6366f1"),
        ("XGBoost",       metrics["xgb"],  "#f59e0b"),
        ("LSTM",          metrics["lstm"], "#22c55e"),
    ]
    ensemble_avg = np.mean([
        metrics["rf"]["cv_accuracy"],
        metrics["xgb"]["cv_accuracy"],
        metrics["lstm"]["cv_accuracy"],
    ])

    for col, (name, m, clr) in zip([m_col1, m_col2, m_col3], model_list):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:{clr}; font-weight:700; margin-bottom:6px">{name}</div>
                <div style="font-size:2rem; font-weight:800; color:white">
                    {m['cv_accuracy']*100:.1f}%
                </div>
                <div style="color:#9ca3af; font-size:0.8rem">
                    ± {m['cv_std']*100:.1f}% std
                </div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:4px">CV Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card" style="border-color:#6366f1">
            <div style="color:#a5b4fc; font-weight:700; margin-bottom:6px">🔗 Ensemble</div>
            <div style="font-size:2rem; font-weight:800; color:white">
                {ensemble_avg*100:.1f}%
            </div>
            <div style="color:#9ca3af; font-size:0.8rem">RF+XGB+LSTM</div>
            <div style="color:#9ca3af; font-size:0.75rem; margin-top:4px">Avg CV Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("")
    fig_bar = go.Figure()
    names  = ["Random Forest", "XGBoost", "LSTM", "Ensemble"]
    accs   = [
        metrics["rf"]["cv_accuracy"],
        metrics["xgb"]["cv_accuracy"],
        metrics["lstm"]["cv_accuracy"],
        ensemble_avg,
    ]
    colors = ["#6366f1", "#f59e0b", "#22c55e", "#a5b4fc"]
    fig_bar.add_trace(go.Bar(
        x=names, y=[a * 100 for a in accs],
        marker_color=colors,
        text=[f"{a*100:.1f}%" for a in accs],
        textposition="outside",
    ))
    fig_bar.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="Random Baseline (50%)")
    fig_bar.update_layout(
        title="Model Accuracy Comparison",
        yaxis_title="CV Accuracy (%)",
        yaxis_range=[0, 100],
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e2130",
        font_color="white",
        height=300,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Feature importance (RF + XGB)
    fi_col1, fi_col2 = st.columns(2)
    for col, (name, m, clr) in zip([fi_col1, fi_col2], model_list[:2]):
        if "feature_importance" in m:
            with col:
                fi_df = pd.DataFrame({
                    "Feature":    FEATURE_COLS,
                    "Importance": m["feature_importance"],
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(
                    fi_df, x="Importance", y="Feature",
                    orientation="h", title=f"{name} — Feature Importance",
                    color_discrete_sequence=[clr],
                )
                fig_fi.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
                    font_color="white", height=300,
                    margin=dict(t=40, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Section 3: Price Chart ────────────────────────────────────────────────
    st.subheader("📉 Price History & Gap Analysis")
    chart_df = feat_df.tail(60).copy()

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["close"],
        mode="lines", name="Close Price",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
    ))
    # Mark gap up / gap down days
    up_days   = chart_df[chart_df["direction"] == 1]
    down_days = chart_df[chart_df["direction"] == 0]
    fig_price.add_trace(go.Scatter(
        x=up_days["date"], y=up_days["close"],
        mode="markers", name="Gap Up",
        marker=dict(color="#22c55e", size=6, symbol="triangle-up"),
    ))
    fig_price.add_trace(go.Scatter(
        x=down_days["date"], y=down_days["close"],
        mode="markers", name="Gap Down",
        marker=dict(color="#ef4444", size=6, symbol="triangle-down"),
    ))
    fig_price.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
        font_color="white", height=350,
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(bgcolor="#1e2130"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Gap % distribution
    fig_gap = px.histogram(
        chart_df, x="gap_pct", nbins=30,
        title="Gap % Distribution (last 60 days)",
        color_discrete_sequence=["#6366f1"],
    )
    fig_gap.add_vline(x=0, line_dash="dash", line_color="white")
    fig_gap.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
        font_color="white", height=250, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.divider()

    # ── Section 4: News Sentiment ─────────────────────────────────────────────
    st.subheader("📰 News Sentiment Analysis (FinBERT)")

    if art_df is not None and not art_df.empty:
        # Sentiment over time
        daily_s = art_df.groupby("date")["score"].mean().reset_index()
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=daily_s["date"],
            y=daily_s["score"],
            marker_color=["#22c55e" if s > 0 else "#ef4444" for s in daily_s["score"]],
            name="Daily Sentiment",
        ))
        fig_sent.add_hline(y=0, line_color="white", line_dash="dash")
        fig_sent.update_layout(
            title="Daily Aggregated Sentiment Score",
            paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
            font_color="white", height=250,
            yaxis_title="Sentiment Score (-1 to +1)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        # Sentiment distribution pie
        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            label_counts = art_df["label"].value_counts()
            fig_pie = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    "positive": "#22c55e",
                    "negative": "#ef4444",
                    "neutral":  "#f59e0b",
                },
            )
            fig_pie.update_layout(
                paper_bgcolor="#0e1117", font_color="white",
                height=280, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with s_col2:
            st.markdown("**Latest Headlines & Scores**")
            display_df = art_df.sort_values("date", ascending=False).head(15)[[
                "date", "headline", "label", "score", "source"
            ]].copy()
            display_df["score"] = display_df["score"].round(3)

            def color_label(val):
                if val == "positive": return "color: #22c55e"
                if val == "negative": return "color: #ef4444"
                return "color: #f59e0b"

            st.dataframe(
                display_df.style.applymap(color_label, subset=["label"]),
                use_container_width=True,
                height=280,
            )
    else:
        st.info("No news data available. Check your FINNHUB_API_KEY in .env")

    st.divider()

    # ── Section 5: How the Ensemble Works ────────────────────────────────────
    st.subheader("🔗 How the Ensemble Works")
    st.markdown(f"""
    Each model sees the same features but learns differently:

    | Model | Type | What it captures | Weight |
    |---|---|---|---|
    | **Random Forest** | ML (Bagging) | Non-linear feature interactions, robust to noise | {WEIGHTS['rf']} |
    | **XGBoost** | ML (Boosting) | Sequential error correction, handles imbalanced data | {WEIGHTS['xgb']} |
    | **LSTM** | Deep Learning | Temporal patterns across {10} trading days | {WEIGHTS['lstm']} |
    | **Ensemble** | Weighted Avg | Reduces individual model errors | — |

    **Final formula:**
    ```
    P(UP) = 0.25 × RF_prob + 0.35 × XGB_prob + 0.40 × LSTM_prob
    ```
    LSTM gets the highest weight because it captures time-series memory that ML models miss.
    """)

else:
    # Landing state
    st.markdown("""
    ## Welcome to Prediction Point 📈

    This system predicts whether a stock will **gap up** or **gap down** tomorrow
    based on today's news sentiment + price action.

    ### How it works:
    1. **Data** — Fetches historical prices (yfinance) + news headlines (Finnhub)
    2. **NLP** — Scores each headline with **FinBERT** (financial BERT model)
    3. **ML Models** — Trains **Random Forest** and **XGBoost** on features
    4. **DL Model** — Trains **LSTM** on 10-day price+sentiment sequences
    5. **Ensemble** — Combines all 3 for the final prediction

    ### Get started:
    1. Add your `FINNHUB_API_KEY` to the `.env` file
    2. Enter a stock symbol in the sidebar (e.g. `AAPL`, `RELIANCE.NS`)
    3. Click **🚀 Run Full Pipeline**
    """)

    st.info("💡 First run downloads FinBERT (~400MB). Subsequent runs are fast.")
