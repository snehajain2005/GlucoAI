"""
GlucoAI Dashboard
Production-grade Streamlit dashboard for glucose monitoring and diabetes risk analysis.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path Setup ──
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "app"))

from app.models.ml_pipeline import (
    DiabetesRiskModel, GlucoseAnomalyDetector,
    GlucoseTrendAnalyzer, MealRecommender,
    generate_synthetic_dataset, FEATURE_DISPLAY_NAMES, FEATURE_COLS
)
from app.utils.helpers import (
    classify_glucose, estimate_hba1c_from_glucose,
    get_glucose_recommendations, activity_level_to_numeric,
    stress_level_to_numeric, generate_risk_explanation,
    generate_glucose_insights, generate_meal_advice
)

# ─────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="GlucoAI · Smart Glucose Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    :root {
        --primary: #00C9A7;
        --primary-dark: #009f84;
        --danger: #FF4757;
        --warning: #FFA502;
        --success: #2ED573;
        --bg-card: rgba(255, 255, 255, 0.05);
        --border: rgba(255, 255, 255, 0.1);
    }
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0A0E1A 0%, #0D1B2E 50%, #091824 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 201, 167, 0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 201, 167, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 { 
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem; 
        font-weight: 700;
        background: linear-gradient(90deg, #00C9A7, #A8EDEA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-header p { color: #8B9AB5; margin: 0.5rem 0 0; font-size: 1rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #0D1B2E, #111827);
        border: 1px solid rgba(0, 201, 167, 0.15);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-label { color: #8B9AB5; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
    .metric-value { font-size: 2rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; color: #E8F4F8; }
    .metric-sub { font-size: 0.8rem; color: #8B9AB5; margin-top: 0.2rem; }
    
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .risk-low { background: rgba(46, 213, 115, 0.15); color: #2ED573; border: 1px solid rgba(46, 213, 115, 0.3); }
    .risk-medium { background: rgba(255, 165, 2, 0.15); color: #FFA502; border: 1px solid rgba(255, 165, 2, 0.3); }
    .risk-high { background: rgba(255, 71, 87, 0.15); color: #FF4757; border: 1px solid rgba(255, 71, 87, 0.3); }
    
    .alert-critical { background: rgba(255, 71, 87, 0.12); border-left: 4px solid #FF4757; padding: 1rem 1.25rem; border-radius: 8px; margin: 0.75rem 0; }
    .alert-warning { background: rgba(255, 165, 2, 0.12); border-left: 4px solid #FFA502; padding: 1rem 1.25rem; border-radius: 8px; margin: 0.75rem 0; }
    .alert-success { background: rgba(46, 213, 115, 0.12); border-left: 4px solid #2ED573; padding: 1rem 1.25rem; border-radius: 8px; margin: 0.75rem 0; }
    
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #E8F4F8;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0, 201, 167, 0.15);
    }
    
    .recommendation-card {
        background: rgba(0, 201, 167, 0.05);
        border: 1px solid rgba(0, 201, 167, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00C9A7, #009f84);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0, 201, 167, 0.3); }
    
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #080D18 0%, #0A1020 100%); border-right: 1px solid rgba(0, 201, 167, 0.1); }
    
    .sidebar-logo { text-align: center; padding: 1.5rem 0 1rem; }
    .sidebar-logo span { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; background: linear-gradient(90deg, #00C9A7, #A8EDEA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "model": None,
        "glucose_history": [],
        "timestamps": [],
        "last_prediction": None,
        "patient_profile": {},
        "demo_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🧠 Loading AI models...")
def load_models():
    model = DiabetesRiskModel()
    model.load()
    recommender = MealRecommender()
    return model, recommender

risk_model, meal_recommender = load_models()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🩺</div>
        <span>GlucoAI</span>
        <div style="color:#8B9AB5; font-size:0.75rem; margin-top:0.25rem">Smart Glucose Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Glucose Tracker", "🔮 Risk Prediction", "🍽️ Meal Advisor", "📈 Trend Analysis", "⚗️ What-If Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**🌐 Mode**")
    rural_mode = st.toggle("Rural / Low-Data Mode", value=False)
    if rural_mode:
        st.info("📡 Simplified UI active — minimal charts, essential data only")

    st.markdown("---")
    st.markdown("**🔑 AI Explanations**")
    groq_key = st.text_input("Groq API Key (optional)", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✓ AI explanations enabled")

    st.markdown("---")
    st.markdown("<div style='color:#8B9AB5;font-size:0.75rem;text-align:center'>GlucoAI v1.0.0<br>Not a substitute for medical advice</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: PLOTLY THEME
# ─────────────────────────────────────────────

CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#8B9AB5", "family": "DM Sans"},
    "xaxis": {"gridcolor": "rgba(255,255,255,0.05)", "linecolor": "rgba(255,255,255,0.1)"},
    "yaxis": {"gridcolor": "rgba(255,255,255,0.05)", "linecolor": "rgba(255,255,255,0.1)"},
}


def make_gauge(value: float, title: str, min_val: float = 0, max_val: float = 1) -> go.Figure:
    color = "#2ED573" if value < 0.35 else "#FFA502" if value < 0.65 else "#FF4757"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={"text": title, "font": {"size": 14, "color": "#8B9AB5"}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B9AB5"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 35], "color": "rgba(46,213,115,0.08)"},
                {"range": [35, 65], "color": "rgba(255,165,2,0.08)"},
                {"range": [65, 100], "color": "rgba(255,71,87,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": value * 100},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10), **{k: v for k, v in CHART_THEME.items() if k in ["paper_bgcolor", "plot_bgcolor", "font"]})
    return fig


# ═══════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🩺 GlucoAI Dashboard</h1>
        <p>AI-powered glucose monitoring · Risk prediction · Personalized insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recent_glucose = st.session_state.glucose_history[-1] if st.session_state.glucose_history else None
        classification = classify_glucose(recent_glucose) if recent_glucose else {"category": "—", "color": "#8B9AB5"}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest Glucose</div>
            <div class="metric-value" style="color:{classification['color']}">{f'{recent_glucose:.0f}' if recent_glucose else '—'} <span style="font-size:1rem">mg/dL</span></div>
            <div class="metric-sub">{classification['category'].replace('_', ' ').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_g = round(np.mean(st.session_state.glucose_history), 1) if st.session_state.glucose_history else None
        est_hba1c = estimate_hba1c_from_glucose(avg_g) if avg_g else None
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Glucose</div>
            <div class="metric-value">{f'{avg_g:.0f}' if avg_g else '—'} <span style="font-size:1rem">mg/dL</span></div>
            <div class="metric-sub">Est. HbA1c: {f'{est_hba1c}%' if est_hba1c else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        readings_count = len(st.session_state.glucose_history)
        anomaly_count = 0
        if st.session_state.glucose_history:
            detector = GlucoseAnomalyDetector()
            anomalies = detector.detect(st.session_state.glucose_history)
            anomaly_count = sum(1 for a in anomalies if a["is_anomaly"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Readings Logged</div>
            <div class="metric-value">{readings_count}</div>
            <div class="metric-sub" style="color:{'#FF4757' if anomaly_count else '#8B9AB5'}">{anomaly_count} anomalies detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        risk_display = st.session_state.last_prediction
        if risk_display:
            risk_pct = risk_display["risk_score"] * 100
            risk_lvl = risk_display["risk_level"]
            risk_color = {"low": "#2ED573", "medium": "#FFA502", "high": "#FF4757"}[risk_lvl]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Diabetes Risk</div>
            <div class="metric-value" style="color:{risk_color if risk_display else '#8B9AB5'}">{f'{risk_pct:.0f}%' if risk_display else '—'}</div>
            <div class="metric-sub">{risk_lvl.upper() + ' RISK' if risk_display else 'Run prediction →'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Load demo data button
    if not st.session_state.demo_loaded:
        st.markdown("<div class='section-header'>Quick Start</div>", unsafe_allow_html=True)
        if st.button("📂 Load Demo Patient Data", use_container_width=True):
            demo_readings = [95, 145, 110, 165, 130, 155, 105, 92, 148, 108, 
                           170, 128, 158, 100, 88, 142, 115, 168, 125, 160]
            now = datetime.now()
            demo_times = [(now - timedelta(hours=i*2)).isoformat() for i in range(len(demo_readings)-1, -1, -1)]
            st.session_state.glucose_history = demo_readings
            st.session_state.timestamps = demo_times
            st.session_state.demo_loaded = True
            st.session_state.patient_profile = {
                "age": 45, "bmi": 26.5, "activity_level": 2,
                "sleep_hours": 7, "stress_level": 1,
                "family_history": 0, "smoker": 0, "hypertension": 0,
            }
            st.success("✅ Demo data loaded! Navigate to other sections to explore.")
            st.rerun()

    # Mini glucose chart if data exists
    if st.session_state.glucose_history and not rural_mode:
        st.markdown("<div class='section-header'>Recent Glucose Trend</div>", unsafe_allow_html=True)
        readings = st.session_state.glucose_history[-20:]
        times = st.session_state.timestamps[-20:] if st.session_state.timestamps else list(range(len(readings)))

        fig = go.Figure()
        # Target range band
        fig.add_hrect(y0=70, y1=180, fillcolor="rgba(0,201,167,0.06)", line_width=0)
        # Lines
        colors = ["#FF4757" if g < 70 or g > 180 else "#00C9A7" for g in readings]
        fig.add_trace(go.Scatter(
            x=list(range(len(readings))),
            y=readings,
            mode="lines+markers",
            name="Glucose",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(color=colors, size=8),
            fill="tozeroy",
            fillcolor="rgba(0,201,167,0.05)",
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", annotation_text="Low 70")
        fig.add_hline(y=180, line_dash="dash", line_color="rgba(255,165,2,0.5)", annotation_text="High 180")
        fig.update_layout(
            height=280, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            **{k: v for k, v in CHART_THEME.items()}
        )
        fig.update_xaxis(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    # Clinical guidelines panel
    st.markdown("<div class='section-header'>📋 Clinical Reference Ranges</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    ranges = [
        ("Fasting Normal", "70–99 mg/dL", "#2ED573"),
        ("Post-Meal Normal", "< 140 mg/dL", "#00C9A7"),
        ("Pre-Diabetes", "100–125 mg/dL", "#FFA502"),
        ("Diabetic Range", "≥ 126 mg/dL", "#FF4757"),
    ]
    for col, (label, val, color) in zip(cols, ranges):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;border-radius:10px;padding:0.8rem;text-align:center">
                <div style="color:{color};font-weight:700;font-size:1.1rem">{val}</div>
                <div style="color:#8B9AB5;font-size:0.75rem;margin-top:0.3rem">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: GLUCOSE TRACKER
# ═══════════════════════════════════════════════

elif page == "📊 Glucose Tracker":
    st.markdown("<h2 style='color:#E8F4F8;font-family:Space Grotesk'>📊 Glucose Tracker</h2>", unsafe_allow_html=True)

    col_input, col_chart = st.columns([1, 2])

    with col_input:
        st.markdown("<div class='section-header'>Log New Reading</div>", unsafe_allow_html=True)
        glucose_val = st.number_input("Glucose Level (mg/dL)", min_value=20.0, max_value=600.0, value=110.0, step=1.0)
        reading_type = st.selectbox("Reading Type", ["random", "fasting", "post_meal", "bedtime"])
        meal_context = st.selectbox("Meal Context", ["N/A", "before_meal", "after_meal"])

        if st.button("➕ Log Reading", use_container_width=True):
            st.session_state.glucose_history.append(glucose_val)
            st.session_state.timestamps.append(datetime.now().isoformat())

            # Instant classification
            cl = classify_glucose(glucose_val, reading_type if reading_type == "fasting" else "random")
            recommendations = get_glucose_recommendations(glucose_val)

            badge_class = {"normal": "alert-success", "hypoglycemia": "alert-critical", "prediabetes": "alert-warning", "diabetes_range": "alert-critical"}.get(cl["category"], "alert-warning")
            st.markdown(f"<div class='{badge_class}'><strong>{cl['category'].replace('_', ' ').title()}</strong> — {glucose_val} mg/dL</div>", unsafe_allow_html=True)
            for r in recommendations[:2]:
                st.markdown(f"• {r}")

        st.markdown("<div class='section-header'>Bulk Import</div>", unsafe_allow_html=True)
        manual_csv = st.text_area("Paste glucose readings (one per line or comma-separated):", height=100)
        if st.button("📥 Import", use_container_width=True) and manual_csv:
            raw = manual_csv.replace(",", "\n").split("\n")
            parsed = [float(x.strip()) for x in raw if x.strip().replace(".", "").isdigit()]
            if parsed:
                now = datetime.now()
                for i, g in enumerate(parsed):
                    st.session_state.glucose_history.append(g)
                    st.session_state.timestamps.append((now - timedelta(hours=len(parsed)-i)).isoformat())
                st.success(f"✅ Imported {len(parsed)} readings")

    with col_chart:
        if st.session_state.glucose_history:
            readings = st.session_state.glucose_history
            times = st.session_state.timestamps if st.session_state.timestamps else list(range(len(readings)))

            # Anomaly detection
            detector = GlucoseAnomalyDetector()
            anomalies = detector.detect(readings)

            # Build chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.08)

            x_vals = list(range(len(readings)))

            # Target range
            fig.add_hrect(y0=70, y1=180, fillcolor="rgba(0,201,167,0.06)", line_width=0, row=1, col=1)

            # Main line
            fig.add_trace(go.Scatter(
                x=x_vals, y=readings,
                mode="lines+markers",
                name="Glucose",
                line=dict(color="#00C9A7", width=2.5),
                marker=dict(
                    color=["#FF4757" if a["is_anomaly"] else "#00C9A7" for a in anomalies],
                    size=[12 if a["is_anomaly"] else 6 for a in anomalies],
                    symbol=["diamond" if a["is_anomaly"] else "circle" for a in anomalies],
                ),
            ), row=1, col=1)

            # Threshold lines
            for y, color, label in [(70, "#FF4757", "Low"), (180, "#FFA502", "High"), (126, "#FF6B81", "Diabetic")]:
                fig.add_hline(y=y, line_dash="dash", line_color=f"{color}60", row=1, col=1)

            # Anomaly scores
            fig.add_trace(go.Bar(
                x=x_vals,
                y=[-a["anomaly_score"] for a in anomalies],
                name="Anomaly Score",
                marker_color=["#FF4757" if a["is_anomaly"] else "#00C9A740" for a in anomalies],
            ), row=2, col=1)

            fig.update_layout(
                height=420, showlegend=True,
                legend=dict(orientation="h", y=1.05),
                **{k: v for k, v in CHART_THEME.items()},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Anomaly alerts
            critical = [a for a in anomalies if a.get("is_critical")]
            if critical:
                for a in critical[:3]:
                    st.markdown(f"<div class='alert-critical'>🚨 <strong>Critical anomaly</strong> at reading #{a['index']+1}: {a['glucose']} mg/dL ({a['anomaly_type']})</div>", unsafe_allow_html=True)
            elif any(a["is_anomaly"] for a in anomalies):
                st.markdown(f"<div class='alert-warning'>⚠️ {sum(1 for a in anomalies if a['is_anomaly'])} anomalies detected in your readings. Review highlighted points above.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-success'>✅ No anomalies detected in your glucose readings.</div>", unsafe_allow_html=True)
        else:
            st.info("📊 No glucose data yet. Log readings above or load demo data from the Dashboard.")


# ═══════════════════════════════════════════════
# PAGE: RISK PREDICTION
# ═══════════════════════════════════════════════

elif page == "🔮 Risk Prediction":
    st.markdown("<h2 style='color:#E8F4F8;font-family:Space Grotesk'>🔮 Diabetes Risk Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9AB5'>Powered by Random Forest + SHAP explainability · Works without glucose data</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown("<div class='section-header'>Patient Profile</div>", unsafe_allow_html=True)
        age = st.slider("Age", 18, 90, 45)
        bmi = st.slider("BMI", 15.0, 50.0, 26.5, 0.1)

        st.markdown("<div class='section-header'>Clinical Data (optional)</div>", unsafe_allow_html=True)
        has_glucose = st.toggle("I have glucose/lab data", value=False)
        glucose_val = hba1c_val = bp_val = insulin_val = None
        if has_glucose:
            glucose_val = st.number_input("Fasting Glucose (mg/dL)", 60.0, 400.0, 100.0)
            hba1c_val = st.number_input("HbA1c (%)", 3.5, 15.0, 5.5, 0.1)
            bp_val = st.number_input("Diastolic BP (mmHg)", 40.0, 130.0, 75.0)
            insulin_val = st.number_input("Insulin (μIU/mL)", 0.0, 400.0, 80.0)

        st.markdown("<div class='section-header'>Lifestyle</div>", unsafe_allow_html=True)
        activity = st.select_slider("Activity Level", options=["Sedentary", "Light", "Moderate", "Active", "Very Active"], value="Moderate")
        sleep = st.slider("Sleep Hours / Night", 3.0, 12.0, 7.0, 0.5)
        stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

        st.markdown("<div class='section-header'>Risk Factors</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        family_hx = c1.checkbox("Family History")
        smoker = c2.checkbox("Smoker")
        hypertension = c3.checkbox("Hypertension")
        pregnancies = st.number_input("Pregnancies (women)", 0, 15, 0)

        predict_btn = st.button("🔮 Predict Risk", use_container_width=True)

    with col_result:
        if predict_btn:
            activity_map = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3, "Very Active": 4}
            stress_map = {"Low": 0, "Moderate": 1, "High": 2}

            features = {
                "age": float(age),
                "bmi": float(bmi),
                "glucose": glucose_val or (100 + bmi * 1.2),
                "hba1c": hba1c_val or (5.0 + bmi * 0.04),
                "blood_pressure": bp_val or 75.0,
                "insulin": insulin_val or 80.0,
                "skin_thickness": 25.0,
                "pregnancies": float(pregnancies),
                "activity_level": float(activity_map[activity]),
                "sleep_hours": float(sleep),
                "stress_level": float(stress_map[stress]),
                "family_history": float(family_hx),
                "smoker": float(smoker),
                "hypertension": float(hypertension),
            }

            with st.spinner("Analyzing risk..."):
                result = risk_model.predict(features)
                st.session_state.last_prediction = result

            # Gauge chart
            fig_gauge = make_gauge(result["risk_score"], "Diabetes Risk Score")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk badge
            level = result["risk_level"]
            badge_cls = f"risk-{level}"
            st.markdown(f"<div style='text-align:center;margin-bottom:1rem'><span class='risk-badge {badge_cls}'>{level.upper()} RISK</span></div>", unsafe_allow_html=True)

            if not rural_mode:
                # SHAP chart
                st.markdown("<div class='section-header'>🧠 SHAP Feature Contribution</div>", unsafe_allow_html=True)
                shap_vals = result["shap_values"]
                sorted_shap = sorted(shap_vals.items(), key=lambda x: x[1])
                labels = [k for k, _ in sorted_shap]
                values = [v for _, v in sorted_shap]
                colors = ["#FF4757" if v > 0 else "#2ED573" for v in values]

                fig_shap = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in values],
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    height=350, title="← Reduces Risk  |  Increases Risk →",
                    title_font_color="#8B9AB5", title_font_size=11,
                    **{k: v for k, v in CHART_THEME.items()},
                    margin=dict(l=10, r=60, t=40, b=10),
                )
                fig_shap.add_vline(x=0, line_color="rgba(255,255,255,0.15)")
                st.plotly_chart(fig_shap, use_container_width=True)

            # AI Explanation
            st.markdown("<div class='section-header'>💬 AI Explanation</div>", unsafe_allow_html=True)
            with st.spinner("Generating explanation..."):
                explanation = generate_risk_explanation(level, result["risk_score"], result["shap_values"], features)
            st.markdown(f"<div class='recommendation-card'><p style='color:#E8F4F8;margin:0;line-height:1.7'>{explanation}</p></div>", unsafe_allow_html=True)

        elif st.session_state.last_prediction:
            result = st.session_state.last_prediction
            fig_gauge = make_gauge(result["risk_score"], "Last Prediction")
            st.plotly_chart(fig_gauge, use_container_width=True)
            level = result["risk_level"]
            st.markdown(f"<div style='text-align:center'><span class='risk-badge risk-{level}'>{level.upper()} RISK</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#8B9AB5'>
                <div style='font-size:3rem;margin-bottom:1rem'>🔮</div>
                <div style='font-size:1.1rem;font-weight:600;color:#E8F4F8'>Fill in the form and run prediction</div>
                <div style='margin-top:0.5rem;font-size:0.9rem'>Works even without glucose data — lifestyle factors alone provide a meaningful estimate</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: MEAL ADVISOR
# ═══════════════════════════════════════════════

elif page == "🍽️ Meal Advisor":
    st.markdown("<h2 style='color:#E8F4F8;font-family:Space Grotesk'>🍽️ Meal Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9AB5'>AI-powered glycemic impact analysis and dietary recommendations</p>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.markdown("<div class='section-header'>Log Your Meal</div>", unsafe_allow_html=True)
        meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack"])
        meal_desc = st.text_area(
            "Describe your meal",
            placeholder="e.g., white rice with chicken curry and a soda",
            height=120
        )

        # Quick meal presets
        st.markdown("**Quick Presets:**")
        preset_cols = st.columns(2)
        presets = {
            "🍳 Healthy Breakfast": "oatmeal with berries and Greek yogurt",
            "🍕 Unhealthy Lunch": "pizza and soda",
            "🥗 Good Dinner": "grilled salmon with broccoli and quinoa",
            "🍔 Fast Food": "burger and french fries",
        }
        for i, (label, meal) in enumerate(presets.items()):
            col = preset_cols[i % 2]
            if col.button(label, use_container_width=True):
                meal_desc = meal

        current_glucose = st.number_input("Current Glucose (optional)", 0.0, 600.0, 0.0, help="0 = not measured")
        analyze_btn = st.button("🔍 Analyze Meal", use_container_width=True)

    with col_result:
        if analyze_btn and meal_desc:
            with st.spinner("Analyzing..."):
                analysis = meal_recommender.analyze_meal(meal_desc, current_glucose or None)

            gi = analysis.get("avg_glycemic_index")
            impact = analysis.get("glucose_impact", "unknown")
            rating = analysis.get("overall_rating", "unknown")

            # Impact indicator
            impact_color = {"low": "#2ED573", "moderate": "#FFA502", "high": "#FF4757", "unknown": "#8B9AB5"}[impact]
            rating_color = {"good": "#2ED573", "moderate": "#FFA502", "poor": "#FF4757", "unknown": "#8B9AB5"}[rating]

            cols = st.columns(3)
            for col, (label, val, color) in zip(cols, [
                ("Glycemic Index", f"{gi:.0f}" if gi else "N/A", "#00C9A7"),
                ("Glucose Impact", impact.title(), impact_color),
                ("Meal Rating", rating.title(), rating_color),
            ]):
                col.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};font-size:1.5rem">{val}</div>
                </div>
                """, unsafe_allow_html=True)

            # GI visual bar
            if gi:
                fig_gi = go.Figure(go.Bar(
                    x=[gi], y=["Your Meal"],
                    orientation="h",
                    marker_color=impact_color,
                    marker_line_width=0,
                ))
                fig_gi.add_vline(x=55, line_dash="dash", line_color="#2ED573", annotation_text="Low GI")
                fig_gi.add_vline(x=70, line_dash="dash", line_color="#FFA502", annotation_text="High GI")
                fig_gi.update_xaxis(range=[0, 100])
                fig_gi.update_layout(height=100, margin=dict(l=0, r=50, t=0, b=0), showlegend=False, **{k: v for k, v in CHART_THEME.items()})
                st.plotly_chart(fig_gi, use_container_width=True)

            # Foods to avoid
            avoid = analysis.get("foods_to_avoid", [])
            alts = analysis.get("safer_alternatives", [])
            safe = analysis.get("safe_foods_in_meal", [])

            if avoid:
                st.markdown("<div class='section-header'>⚠️ High-GI Foods Detected</div>", unsafe_allow_html=True)
                for food in avoid:
                    st.markdown(f"<div class='alert-warning'>❌ <strong>{food}</strong> — high glycemic impact</div>", unsafe_allow_html=True)

            if alts:
                st.markdown("<div class='section-header'>✅ Healthier Alternatives</div>", unsafe_allow_html=True)
                for alt in alts:
                    st.markdown(f"<div class='alert-success'>✅ {alt}</div>", unsafe_allow_html=True)

            if safe:
                st.markdown(f"<p style='color:#8B9AB5;font-size:0.85rem'>✓ Diabetes-friendly in your meal: {', '.join(safe)}</p>", unsafe_allow_html=True)

            # AI advice
            st.markdown("<div class='section-header'>💬 Dietitian AI Advice</div>", unsafe_allow_html=True)
            with st.spinner("Generating advice..."):
                ai_advice = generate_meal_advice(meal_desc, analysis, current_glucose or None)
            st.markdown(f"<div class='recommendation-card'><p style='color:#E8F4F8;margin:0;line-height:1.7'>{ai_advice}</p></div>", unsafe_allow_html=True)

        else:
            st.info("Enter a meal description above to get glycemic analysis and recommendations.")


# ═══════════════════════════════════════════════
# PAGE: TREND ANALYSIS
# ═══════════════════════════════════════════════

elif page == "📈 Trend Analysis":
    st.markdown("<h2 style='color:#E8F4F8;font-family:Space Grotesk'>📈 Trend Analysis</h2>", unsafe_allow_html=True)

    if not st.session_state.glucose_history:
        st.warning("No glucose data available. Log readings in the Glucose Tracker or load demo data.")
    else:
        readings = st.session_state.glucose_history
        times = st.session_state.timestamps

        # Key stats
        tir = GlucoseTrendAnalyzer.time_in_range(readings)
        trend_df = GlucoseTrendAnalyzer.rolling_stats(readings, times)
        direction = GlucoseTrendAnalyzer.trend_direction(readings)

        # TIR donut chart
        col_donut, col_stats = st.columns([1, 1.5])
        with col_donut:
            fig_tir = go.Figure(go.Pie(
                values=[tir["in_range_pct"], tir["below_range_pct"], tir["above_range_pct"]],
                labels=["In Range", "Below Range", "Above Range"],
                hole=0.65,
                marker_colors=["#00C9A7", "#FF4757", "#FFA502"],
                textinfo="label+percent",
            ))
            fig_tir.update_layout(
                title="Time in Range", height=280,
                annotations=[dict(text=f"{tir['in_range_pct']}%", x=0.5, y=0.5, font_size=22, showarrow=False, font_color="#E8F4F8")],
                **{k: v for k, v in CHART_THEME.items() if k in ["paper_bgcolor", "plot_bgcolor", "font"]},
                showlegend=True, legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_tir, use_container_width=True)

        with col_stats:
            st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
            stat_data = {
                "Mean": f"{tir['mean']} mg/dL",
                "Std Dev": f"{tir['std']} mg/dL",
                "Coefficient of Variation": f"{tir['cv']}%",
                "Trend Direction": direction.title(),
                "Est. HbA1c": f"{estimate_hba1c_from_glucose(tir['mean'])}%",
                "Total Readings": str(len(readings)),
            }
            for label, val in stat_data.items():
                direction_color = "#2ED573" if "improv" in val.lower() else "#FF4757" if "wors" in val.lower() else "#E8F4F8"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                    <span style="color:#8B9AB5">{label}</span>
                    <span style="color:{direction_color};font-weight:600">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        # Rolling average chart
        if not rural_mode:
            st.markdown("<div class='section-header'>Rolling Average & Variability</div>", unsafe_allow_html=True)
            x = list(range(len(readings)))

            fig_roll = go.Figure()
            fig_roll.add_hrect(y0=70, y1=180, fillcolor="rgba(0,201,167,0.05)", line_width=0)

            # Confidence band
            upper = (trend_df["rolling_mean"] + trend_df["rolling_std"]).tolist()
            lower = (trend_df["rolling_mean"] - trend_df["rolling_std"]).tolist()
            fig_roll.add_trace(go.Scatter(x=x+x[::-1], y=upper+lower[::-1], fill="toself", fillcolor="rgba(0,201,167,0.08)", line_width=0, name="±1 SD"))

            # Raw readings (faint)
            fig_roll.add_trace(go.Scatter(x=x, y=readings, mode="markers", marker=dict(color="#00C9A730", size=5), name="Raw"))

            # Rolling mean
            fig_roll.add_trace(go.Scatter(
                x=x, y=trend_df["rolling_mean"].tolist(),
                mode="lines", line=dict(color="#00C9A7", width=2.5),
                name="Rolling Mean"
            ))

            # Spikes
            spike_idx = [i for i, s in enumerate(trend_df["is_spike"].tolist()) if s]
            spike_vals = [readings[i] for i in spike_idx]
            if spike_idx:
                fig_roll.add_trace(go.Scatter(
                    x=spike_idx, y=spike_vals, mode="markers",
                    marker=dict(symbol="triangle-up", size=14, color="#FFA502"),
                    name=f"Spikes ({len(spike_idx)})"
                ))

            fig_roll.update_layout(height=350, **{k: v for k, v in CHART_THEME.items()}, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_roll, use_container_width=True)

            # AI insights
            st.markdown("<div class='section-header'>💬 AI Trend Insights</div>", unsafe_allow_html=True)
            detector = GlucoseAnomalyDetector()
            anomalies = detector.detect(readings)
            with st.spinner("Generating insights..."):
                insights = generate_glucose_insights(readings, anomalies, direction, tir)
            st.markdown(f"<div class='recommendation-card'><p style='color:#E8F4F8;margin:0;line-height:1.7'>{insights}</p></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════

elif page == "⚗️ What-If Analysis":
    st.markdown("<h2 style='color:#E8F4F8;font-family:Space Grotesk'>⚗️ What-If Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9AB5'>Simulate diet or lifestyle changes and see projected improvements</p>", unsafe_allow_html=True)

    tab_meal, tab_lifestyle = st.tabs(["🍽️ Meal Swap", "🏃 Lifestyle Simulation"])

    with tab_meal:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Current Meal</div>", unsafe_allow_html=True)
            current_meal = st.text_area("What do you currently eat?", placeholder="e.g., white rice with fried chicken and soda", height=100, key="current")
        with col2:
            st.markdown("<div class='section-header'>Proposed Alternative</div>", unsafe_allow_html=True)
            proposed_meal = st.text_area("What would you switch to?", placeholder="e.g., brown rice with grilled chicken and water", height=100, key="proposed")

        if st.button("🔬 Simulate Swap", use_container_width=True) and current_meal and proposed_meal:
            result = meal_recommender.what_if_analysis(current_meal, proposed_meal)

            c_gi = result["current_meal"].get("avg_glycemic_index") or 65
            p_gi = result["proposed_meal"].get("avg_glycemic_index") or 55
            gi_diff = result["gi_difference"]

            col_c, col_arrow, col_p = st.columns([2, 0.5, 2])
            with col_c:
                color = "#FF4757" if c_gi > 65 else "#FFA502" if c_gi > 50 else "#2ED573"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-label">Current GI</div>
                    <div class="metric-value" style="color:{color}">{c_gi:.0f}</div>
                    <div class="metric-sub">{result['current_meal'].get('glucose_impact','?')} impact</div>
                </div>
                """, unsafe_allow_html=True)
            with col_arrow:
                st.markdown("<div style='text-align:center;padding:2rem 0;font-size:2rem'>→</div>", unsafe_allow_html=True)
            with col_p:
                color = "#FF4757" if p_gi > 65 else "#FFA502" if p_gi > 50 else "#2ED573"
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-label">Proposed GI</div>
                    <div class="metric-value" style="color:{color}">{p_gi:.0f}</div>
                    <div class="metric-sub">{result['proposed_meal'].get('glucose_impact','?')} impact</div>
                </div>
                """, unsafe_allow_html=True)

            # Impact summary
            if gi_diff > 0:
                st.markdown(f"<div class='alert-success'>✅ <strong>Improvement: GI reduced by {gi_diff:.0f} points</strong> — projected glucose reduction ~{result['projected_glucose_change']:.0f} mg/dL. {result['recommendation']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-warning'>⚠️ <strong>Similar or higher GI</strong> — the proposed meal has similar or higher glycemic impact. Consider more low-GI options.</div>", unsafe_allow_html=True)

    with tab_lifestyle:
        st.markdown("<div class='section-header'>Simulate Lifestyle Changes</div>", unsafe_allow_html=True)
        st.markdown("Adjust sliders to see how changes affect your risk score")

        if st.session_state.last_prediction:
            baseline = st.session_state.last_prediction["risk_score"]
            
            col_sliders, col_result = st.columns(2)
            with col_sliders:
                d_activity = st.slider("Activity Level Change", -2, 2, 0, help="Negative = less active, Positive = more active")
                d_sleep = st.slider("Sleep Change (hours)", -3.0, 3.0, 0.0, 0.5)
                d_bmi = st.slider("BMI Change", -10.0, 5.0, 0.0, 0.5)
                quit_smoking = st.checkbox("Quit Smoking")
                control_bp = st.checkbox("Control Hypertension")

            with col_result:
                # Rough approximation of impact
                impact = -(d_activity * 0.04) - (d_sleep * 0.015) - (d_bmi * 0.01) - (0.1 if quit_smoking else 0) - (0.08 if control_bp else 0)
                new_risk = max(0, min(1, baseline + impact))

                fig = make_gauge(new_risk, "Projected Risk After Changes")
                st.plotly_chart(fig, use_container_width=True)

                delta = baseline - new_risk
                if delta > 0:
                    st.markdown(f"<div class='alert-success'>✅ These changes could reduce your risk by <strong>{delta:.0%}</strong></div>", unsafe_allow_html=True)
                elif delta < 0:
                    st.markdown(f"<div class='alert-warning'>⚠️ These changes may increase risk by <strong>{abs(delta):.0%}</strong></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-success'>No significant change projected</div>", unsafe_allow_html=True)
        else:
            st.info("Run a risk prediction first to enable lifestyle simulation.")

# Footer
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#8B9AB5;font-size:0.8rem;border-top:1px solid rgba(255,255,255,0.05);margin-top:2rem'>
    GlucoAI v1.0.0 · For educational and research purposes only · Not a substitute for medical advice<br>
    Built with FastAPI · Streamlit · Scikit-learn · SHAP · Plotly
</div>
""", unsafe_allow_html=True)
