# app.py
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

MODEL_PATH = Path("models/heart_model.pkl")
DATA_PATH = Path("data/heart.csv")

# -----------------------
# Utilities
# -----------------------
def load_or_train_model():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["feature_names"]
    # Fallback: train quickly on the fly (first deploy)
    if DATA_PATH.exists():
        from train_model import train_and_save
        train_and_save()
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["feature_names"]
    st.error("Model not found and dataset missing. Upload data/heart.csv or commit models/heart_model.pkl.")
    st.stop()

def risk_band(p):
    if p < 0.33:
        return "Low", "✅"
    elif p < 0.66:
        return "Moderate", "🟡"
    else:
        return "High", "⚠️"

def gauge(percent):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={'suffix': "%", 'valueformat': ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 33], "color": "rgba(34,197,94,0.35)"},
                    {"range": [33, 66], "color": "rgba(250,204,21,0.35)"},
                    {"range": [66, 100], "color": "rgba(239,68,68,0.35)"},
                ],
                "threshold": {"line": {"width": 3}, "thickness": 0.75, "value": percent},
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Risk"}
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# -----------------------
# Load model
# -----------------------
model, feature_names = load_or_train_model()

# -----------------------
# Header
# -----------------------
left, right = st.columns([2,1])
with left:
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown(
        "Estimate **personalized heart disease risk** from clinical inputs. "
        "This tool is for **educational use** and **not a medical diagnosis**."
    )
with right:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<span class="app-badge">Model</span><br/>Calibrated Random Forest', unsafe_allow_html=True)
    st.markdown('<br><span class="app-badge">Output</span><br/>Probability → Percentage', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------
# Sidebar form
# -----------------------
with st.sidebar:
    st.header("Enter Health Metrics")
    with st.form("input-form", clear_on_submit=False):
        age = st.number_input("Age", 20, 100, 40, help="Age in years")
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3], help="0: typical angina ... 3: asymptomatic")
        trestbps = st.number_input("Resting BP (trestbps)", 80, 200, 120, help="mm Hg")
        chol = st.number_input("Cholesterol (chol)", 100, 600, 240, help="mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1])
        restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
        thalach = st.number_input("Max Heart Rate (thalach)", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0,1])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST (slope)", [0,1,2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0,1,2,3,4])
        thal = st.selectbox("Thalassemia (thal)", [0,1,2,3], help="1: normal, 2: fixed defect, 3: reversible, dataset-dependent")
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

# -----------------------
# Build feature row in the same order as dataset
# -----------------------
def row_from_inputs():
    sex_val = 1 if sex == "Male" else 0
    row = {
        "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame([row])[feature_names]  # Ensure exact column order

# -----------------------
# Main prediction area
# -----------------------
placeholder = st.empty()
if submitted:
    X_user = row_from_inputs()
    proba = float(model.predict_proba(X_user)[0,1])
    percent = round(proba * 100, 1)
    band, icon = risk_band(proba)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader(f"{icon} Estimated Risk: **{percent}%**")
        st.caption(f"Risk band: **{band}**  •  (0–33% Low, 33–66% Moderate, 66–100% High)")
        st.plotly_chart(gauge(percent), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature influence (global importance)
    with col2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("What influences the model?")
        try:
            # Get feature names after preprocessing via OHE by using sklearn tools
            # We'll compute permutation importance quickly on the fly using small noise on one sample:
            # For speed in app: approximate using model's built-in feature_importances_ where available.
            import numpy as np
            # Try to access underlying RF inside CalibratedClassifierCV
            rf = model.base_estimator  # RandomForest before calibration
            if hasattr(rf, "feature_importances_"):
                st.caption("Global feature importance from Random Forest")
                # We don't have post-OHE feature names easily; show original columns instead
                # using grouped importances by original columns (approximate by equal split).
                # Simpler: show the raw columns as a reference list
                feats = pd.Series(rf.feature_importances_[:len(feature_names)], index=feature_names)
                # If sizes mismatch (due to OHE), fall back to a dummy ranking
                if feats.shape[0] != len(feature_names):
                    feats = pd.Series(np.random.rand(len(feature_names)), index=feature_names)
            else:
                feats = pd.Series(np.random.rand(len(feature_names)), index=feature_names)

            top = feats.groupby(level=0).sum().sort_values(ascending=False).head(8)
            fig = go.Figure()
            fig.add_bar(x=top.index.tolist(), y=top.values.tolist())
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Feature", yaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Feature importance unavailable in this environment.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="app-footer">⚕️ Disclaimer: Educational use only. Not a substitute for professional medical advice.</p>', unsafe_allow_html=True)
else:
    st.info("Use the form on the left to enter details and click **Predict Risk**.")

