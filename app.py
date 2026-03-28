import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.features   import load_dataset, get_train_test_split, FEATURE_COLS
from src.models.ensemble  import train_all_models, predict_ensemble, WEIGHTS

st.set_page_config(page_title="Prediction Point", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border-radius: 12px;
        padding: 20px; text-align: center; border: 1px solid #2d3250;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Prediction Point")
    st.caption("NIFTY Gap Prediction · LR + XGBoost + LSTM")
    st.divider()
    st.markdown("**Dataset**")
    st.caption("NIFTY 50 · 2015–2026 · Global Index Signals")
    st.divider()
    st.markdown("**Model Weights (Ensemble)**")
    st.caption(f"LR: {WEIGHTS['lr']} · XGB: {WEIGHTS['xgb']} · LSTM: {WEIGHTS['lstm']}")
    st.divider()
    run = st.button("Run Pipeline", type="primary", use_container_width=True)
    st.caption("Loads dataset → trains 3 models → compares → predicts")

for key in ["trained", "metrics", "prediction", "df", "test_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Pipeline ───────────────────────────────────────────────────────────────────
if run:
    st.session_state.trained = False

    with st.status("Loading dataset...", expanded=True):
        df = load_dataset()
        st.write(f"Loaded {len(df)} rows | Features: {FEATURE_COLS}")
        st.session_state.df = df

    with st.status("Splitting data (80/20 time-based)...", expanded=True):
        X_train, X_test, y_train, y_test, train_df, test_df = get_train_test_split(df)
        st.write(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        st.write(f"Train period: {train_df['Date'].iloc[0].date()} to {train_df['Date'].iloc[-1].date()}")
        st.write(f"Test period:  {test_df['Date'].iloc[0].date()} to {test_df['Date'].iloc[-1].date()}")
        st.session_state.test_df = test_df

    with st.status("Training Logistic Regression + XGBoost + LSTM...", expanded=True):
        try:
            metrics = train_all_models(X_train, y_train, X_test, y_test)
            st.write(f"Logistic Regression — Accuracy: **{metrics['lr']['accuracy']*100:.1f}%**")
            st.write(f"XGBoost             — Accuracy: **{metrics['xgb']['accuracy']*100:.1f}%**")
            st.write(f"LSTM                — Accuracy: **{metrics['lstm']['accuracy']*100:.1f}%**")
            st.session_state.metrics = metrics
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    with st.status("Generating ensemble prediction...", expanded=True):
        X_all = df[FEATURE_COLS].values.astype("float32")
        prediction = predict_ensemble(X_all)
        st.write(f"Prediction: **{prediction['direction']}** | Confidence: {prediction['confidence']*100:.1f}%")
        st.session_state.prediction = prediction
        st.session_state.trained = True

# ── Dashboard ──────────────────────────────────────────────────────────────────
if st.session_state.trained:
    pred    = st.session_state.prediction
    metrics = st.session_state.metrics
    df      = st.session_state.df
    test_df = st.session_state.test_df

    st.divider()

    # ── Prediction ────────────────────────────────────────────────────────────
    st.subheader("Tomorrow's Prediction")
    is_up  = pred["direction"] == "GAP UP"
    color  = "#22c55e" if is_up else "#ef4444"
    arrow  = "▲" if is_up else "▼"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af; font-size:0.9rem; margin-bottom:8px">NIFTY 50 · Next Day Gap</div>
            <div style="color:{color}; font-size:3rem; font-weight:900">{arrow} {pred['direction']}</div>
            <div style="color:#e5e7eb; font-size:1.1rem; margin-top:8px">
                Ensemble Confidence: <b>{pred['confidence']*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred["ensemble_prob"] * 100,
            title={"text": "UP Probability", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, 50],   "color": "#2d1b1b"},
                    {"range": [50, 100], "color": "#1b2d1b"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 50},
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig_gauge.update_layout(height=220, paper_bgcolor="#0e1117", font_color="white",
                                margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown("**Individual Model Votes**")
        for name, prob in [("Logistic", pred["lr_prob"]), ("XGBoost", pred["xgb_prob"]), ("LSTM", pred["lstm_prob"])]:
            vote_color = "#22c55e" if prob >= 0.5 else "#ef4444"
            vote = "▲ UP" if prob >= 0.5 else "▼ DOWN"
            st.markdown(f"**{name}** — <span style='color:{vote_color}'>{vote}</span> ({prob*100:.1f}%)",
                        unsafe_allow_html=True)
            st.progress(float(prob))

    st.divider()

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.subheader("Model Performance Comparison")
    st.caption("Time-based 80/20 split — no data leakage")

    model_list = [
        ("Logistic Regression", metrics["lr"],   "#6366f1"),
        ("XGBoost",             metrics["xgb"],  "#f59e0b"),
        ("LSTM",                metrics["lstm"], "#22c55e"),
    ]
    ensemble_acc = np.mean([m["accuracy"] for _, m, _ in model_list])

    c1, c2, c3, c4 = st.columns(4)
    for col, (name, m, clr) in zip([c1, c2, c3], model_list):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:{clr}; font-weight:700; margin-bottom:6px">{name}</div>
                <div style="font-size:2rem; font-weight:800; color:white">{m['accuracy']*100:.1f}%</div>
                <div style="color:#9ca3af; font-size:0.75rem; margin-top:4px">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-color:#6366f1">
            <div style="color:#a5b4fc; font-weight:700; margin-bottom:6px">Ensemble</div>
            <div style="font-size:2rem; font-weight:800; color:white">{ensemble_acc*100:.1f}%</div>
            <div style="color:#9ca3af; font-size:0.75rem; margin-top:4px">Avg Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart
    fig_bar = go.Figure()
    names  = ["Logistic", "XGBoost", "LSTM", "Ensemble"]
    accs   = [metrics["lr"]["accuracy"], metrics["xgb"]["accuracy"], metrics["lstm"]["accuracy"], ensemble_acc]
    colors = ["#6366f1", "#f59e0b", "#22c55e", "#a5b4fc"]
    fig_bar.add_trace(go.Bar(x=names, y=[a*100 for a in accs], marker_color=colors,
                             text=[f"{a*100:.1f}%" for a in accs], textposition="outside"))
    fig_bar.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Random Baseline (50%)")
    fig_bar.update_layout(yaxis_title="Accuracy (%)", yaxis_range=[0, 100],
                          paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
                          font_color="white", height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Feature Importance (XGBoost) ──────────────────────────────────────────
    st.subheader("Feature Importance (XGBoost)")
    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": metrics["xgb"]["feature_importance"],
    }).sort_values("Importance", ascending=True)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=["#f59e0b"])
    fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
                         font_color="white", height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Gap % Distribution ────────────────────────────────────────────────────
    st.subheader("Gap % Distribution (Full Dataset)")
    fig_gap = px.histogram(df, x="Gap %", nbins=50, color_discrete_sequence=["#6366f1"])
    fig_gap.add_vline(x=0, line_dash="dash", line_color="white",
                      annotation_text="0% (Gap Up / Down boundary)")
    fig_gap.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1e2130",
                          font_color="white", height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig_gap, use_container_width=True)

    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────────────
    st.subheader("Analysis")
    best = max(model_list, key=lambda x: x[1]["accuracy"])
    st.markdown(f"""
    **Best model: {best[0]}** with **{best[1]['accuracy']*100:.1f}%** test accuracy.

    | Model | Type | Strength |
    |---|---|---|
    | Logistic Regression | Linear baseline | Interpretable, fast |
    | XGBoost | Gradient Boosting | Best for tabular data, nonlinear patterns |
    | LSTM | Deep Learning | Temporal sequence memory (5-day window) |

    **Conclusion:** Tree-based models (XGBoost) outperform linear and deep learning models
    on structured tabular features with limited dataset size. LSTM requires more data to
    capture meaningful temporal patterns.

    **Ensemble formula:**
    ```
    P(UP) = {WEIGHTS['lr']} x LR + {WEIGHTS['xgb']} x XGB + {WEIGHTS['lstm']} x LSTM
    ```
    """)

else:
    st.markdown("""
    ## Prediction Point

    Predicts whether **NIFTY 50 will gap up or gap down** at next market open
    using global index signals.

    ### Pipeline:
    1. **Dataset** — NIFTY 50 (2015–2026) with SP500, Nasdaq, Nikkei, Hang Seng signals
    2. **Split** — 80% train / 20% test (time-based, no leakage)
    3. **Models** — Logistic Regression (baseline) + XGBoost + LSTM
    4. **Evaluation** — Accuracy + Confusion Matrix + Feature Importance
    5. **Ensemble** — Weighted prediction for final signal

    ### Features:
    | Feature | What it represents |
    |---|---|
    | SP500 % | Global sentiment |
    | Nasdaq % | Tech / risk appetite |
    | Nikkei % | Asia sentiment (prev day) |
    | HangSeng % | Asia sentiment (prev day) |
    | NIFTY Prev % | Local momentum |
    | Volatility | Market regime (5-day rolling std) |

    Click **Run Pipeline** to start.
    """)
