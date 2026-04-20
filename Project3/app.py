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
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "downtime_model.pkl")
    return joblib.load(model_path)

df = load_data()
model = load_model()

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
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🔧 Machine Inputs")

air_temp = st.sidebar.slider("Air Temperature",
    float(df['Air temperature [K]'].min()),
    float(df['Air temperature [K]'].max()))

process_temp = st.sidebar.slider("Process Temperature",
    float(df['Process temperature [K]'].min()),
    float(df['Process temperature [K]'].max()))

rpm = st.sidebar.slider("Rotational Speed",
    int(df['Rotational speed [rpm]'].min()),
    int(df['Rotational speed [rpm]'].max()))

torque = st.sidebar.slider("Torque",
    float(df['Torque [Nm]'].min()),
    float(df['Torque [Nm]'].max()))

tool_wear = st.sidebar.slider("Tool Wear",
    int(df['Tool wear [min]'].min()),
    int(df['Tool wear [min]'].max()))

type_map = {'L': 0, 'M': 1, 'H': 2}
type_input = st.sidebar.selectbox("Machine Type", list(type_map.keys()))
machine_type = type_map[type_input]

selected_types = st.sidebar.multiselect(
    "Machine Type Filter",
    list(type_map.keys()),
    default=list(type_map.keys())
)

selected_failure = st.sidebar.multiselect(
    "Failure Status",
    [0, 1],
    default=[0, 1],
    format_func=lambda x: "Failure" if x == 1 else "No Failure"
)

selected_palette = st.sidebar.selectbox(
    "Chart Color",
    ["blues", "viridis", "plasma", "matter"]
)

shap_sample_size = st.sidebar.slider(
    "SHAP sample size",
    50,
    min(500, len(df)),
    min(200, len(df)),
    50
)

# =========================
# INPUT DATA
# =========================
temp_diff = process_temp - air_temp
load = torque * rpm

input_df = pd.DataFrame([[
    air_temp, process_temp, rpm, torque,
    tool_wear, machine_type, temp_diff, load
]], columns=feature_cols)

filtered_df = df[
    df['Type'].isin([type_map[t] for t in selected_types]) &
    df['Machine failure'].isin(selected_failure)
].copy()

if filtered_df.empty:
    st.warning("No data for selected filters. Showing full dataset.")
    filtered_df = df.copy()

filtered_df['Failure_label'] = filtered_df['Machine failure'].map({
    0: 'No Failure',
    1: 'Failure'
})

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Real-Time Prediction")

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("⚠ High Risk")
    else:
        st.success("✅ Low Risk")

with col2:
    st.metric("Failure Probability", f"{prob:.2%}")

# =========================
# GAUGE
# =========================
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob * 100,
    number={'suffix': '%'},
    title={'text': "Downtime Risk"},
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
# INSIGHTS
# =========================
st.subheader("📊 Insights")

st.metric("Records", len(filtered_df))
st.metric("Failure Rate", f"{filtered_df['Machine failure'].mean()*100:.1f}%")

fig1 = px.histogram(filtered_df, x='Failure_label', color='Type_label')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(filtered_df, x='Failure_label', y='Torque [Nm]', color='Failure_label')
st.plotly_chart(fig2, use_container_width=True)

# =========================
# CORRELATION
# =========================
corr = filtered_df[feature_cols + ['Machine failure']].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale=selected_palette
)
st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# SHAP (ROBUST FIX)
# =========================
st.subheader("🧠 SHAP Explainability")

try:
    explainer = shap.TreeExplainer(model)
    X_sample = df[feature_cols].sample(min(shap_sample_size, len(df)))

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]

    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values
    else:
        shap_vals = np.array(shap_values)

    shap_vals = np.squeeze(shap_vals)
    importance = np.mean(np.abs(shap_vals), axis=0)
    importance = np.array(importance).flatten()

    shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance')

    fig_shap = px.bar(
        shap_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=selected_palette,
        title="SHAP Feature Importance"
    )

    st.plotly_chart(fig_shap, use_container_width=True)

except Exception as e:
    st.warning(f"SHAP error: {e}")

# =========================
# SAMPLE
# =========================
st.subheader("🧪 Sample Predictions")

sample = df.sample(5)
sample['Prediction'] = model.predict(sample[feature_cols])

st.dataframe(sample)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Smart Manufacturing Downtime Prediction System")
