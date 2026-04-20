import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Downtime Risk Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data1.csv")

    df = pd.read_csv(file_path)

    df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')

    df['Type'] = df['Type'].replace({'L': 0, 'M': 1, 'H': 2})
    df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
    df['Type_label'] = df['Type'].map({0: 'Low', 1: 'Medium', 2: 'High'})

    df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Load'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

    return df

# =========================
# LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(base_dir, "downtime_model.pkl"))

    scaler_path = os.path.join(base_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler

df = load_data()
model, scaler = load_artifacts()

# =========================
# FEATURES
# =========================
feature_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type',
    'Temp_diff',
    'Load'
]

# =========================
# TITLE
# =========================
st.title("🏭 Smart Manufacturing Downtime Risk Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔧 Machine Inputs")

def safe_slider(label, series):
    return st.sidebar.slider(
        label,
        float(series.min()),
        float(series.max()),
        float(series.mean())
    )

air_temp = safe_slider("Air Temperature", df['Air temperature [K]'])
process_temp = safe_slider("Process Temperature", df['Process temperature [K]'])
rpm = st.sidebar.slider("Rotational Speed",
                        int(df['Rotational speed [rpm]'].min()),
                        int(df['Rotational speed [rpm]'].max()),
                        int(df['Rotational speed [rpm]'].mean()))

torque = safe_slider("Torque", df['Torque [Nm]'])

tool_wear = st.sidebar.slider("Tool Wear",
                              int(df['Tool wear [min]'].min()),
                              int(df['Tool wear [min]'].max()),
                              int(df['Tool wear [min]'].mean()))

type_map = {'L': 0, 'M': 1, 'H': 2}
type_input = st.sidebar.selectbox("Machine Type", list(type_map.keys()))
machine_type = type_map[type_input]

# =========================
# INPUT FEATURES
# =========================
temp_diff = process_temp - air_temp
load = torque * rpm

input_df = pd.DataFrame([[ 
    air_temp, process_temp, rpm, torque,
    tool_wear, machine_type, temp_diff, load
]], columns=feature_cols)

# =========================
# ⚠ OUT-OF-DISTRIBUTION CHECK
# =========================
if (
    (input_df < df[feature_cols].min()).any(axis=None) or
    (input_df > df[feature_cols].max()).any(axis=None)
):
    st.warning("⚠ Input values are outside training data range — prediction may be unreliable.")

# =========================
# APPLY SCALING (CRITICAL FIX)
# =========================
def transform(X):
    if scaler:
        return scaler.transform(X)
    return X

X_input = transform(input_df)
X_all = transform(df[feature_cols])

# =========================
# MODEL PREDICTION
# =========================
st.subheader("🔮 Real-Time Prediction")

prediction = model.predict(X_input)[0]
prob = model.predict_proba(X_input)[0][1]

col1, col2 = st.columns(2)

with col1:
    if prob > 0.7:
        st.error("⚠ High Risk of Downtime")
    elif prob > 0.3:
        st.warning("⚠ Medium Risk")
    else:
        st.success("✅ Low Risk of Downtime")

with col2:
    st.metric("Current Failure Probability", f"{prob:.2%}")

# =========================
# DATA FILTER
# =========================
filtered_df = df.copy()
real_failure_rate = filtered_df['Machine failure'].mean() * 100

model_risk_avg = model.predict_proba(transform(filtered_df[feature_cols]))[:, 1].mean() * 100

# =========================
# SYSTEM INSIGHTS
# =========================
st.subheader("📊 System Insights")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(filtered_df))
c2.metric("Actual Failure Rate", f"{real_failure_rate:.2f}%")
c3.metric("Avg Model Risk", f"{model_risk_avg:.2f}%")

# =========================
# GAUGE
# =========================
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=prob * 100,
    delta={'reference': model_risk_avg},
    number={'suffix': '%'},
    title={'text': "Current Risk vs Dataset Average"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': 'green'},
            {'range': [30, 70], 'color': 'orange'},
            {'range': [70, 100], 'color': 'red'}
        ]
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)

# =========================
# DISTRIBUTION
# =========================
fig1 = px.histogram(
    df,
    x='Machine failure',
    color='Machine failure',
    title="Failure Distribution"
)
st.plotly_chart(fig1, use_container_width=True)

# =========================
# SHAP
# =========================
st.subheader("🧠 SHAP Explainability")

try:
    explainer = shap.TreeExplainer(model)
    sample = df[feature_cols].sample(min(200, len(df)))
    sample_transformed = transform(sample)

    shap_values = explainer.shap_values(sample_transformed)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    importance = np.mean(np.abs(shap_vals), axis=0)

    shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance')

    fig_shap = px.bar(
        shap_df,
        x='Importance',
        y='Feature',
        orientation='h'
    )

    st.plotly_chart(fig_shap, use_container_width=True)

except Exception as e:
    st.warning(f"SHAP error: {e}")

# =========================
# SAMPLE DATA
# =========================
st.subheader("🧪 Sample Predictions")

sample = df.sample(5)
sample['Prediction'] = model.predict(transform(sample[feature_cols]))

st.dataframe(sample)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Smart Manufacturing Downtime Prediction System")
