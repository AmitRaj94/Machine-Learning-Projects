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
# LOAD DATA (FIXED)
# =========================
@st.cache_data
def load_data():
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data1.csv")

    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')

    # 🔥 FIX 1: Encode Type properly (PERMANENT FIX)
    df['Type'] = df['Type'].replace({'L': 0, 'M': 1, 'H': 2})
    df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
    df['Type_label'] = df['Type'].map({0: 'Low', 1: 'Medium', 2: 'High'})

    # 🔥 FIX 2: Feature Engineering
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
# FEATURE LIST (MUST MATCH TRAINING)
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
st.title("🏭 Smart Manufacturing Downtime Risk Pridiction Dashboard")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("🔧 Machine Inputs")

air_temp = st.sidebar.slider("Air Temperature", float(df['Air temperature [K]'].min()), float(df['Air temperature [K]'].max()))
process_temp = st.sidebar.slider("Process Temperature", float(df['Process temperature [K]'].min()), float(df['Process temperature [K]'].max()))
rpm = st.sidebar.slider("Rotational Speed", int(df['Rotational speed [rpm]'].min()), int(df['Rotational speed [rpm]'].max()))
torque = st.sidebar.slider("Torque", float(df['Torque [Nm]'].min()), float(df['Torque [Nm]'].max()))
tool_wear = st.sidebar.slider("Tool Wear", int(df['Tool wear [min]'].min()), int(df['Tool wear [min]'].max()))

type_map = {'L': 0, 'M': 1, 'H': 2}
type_input = st.sidebar.selectbox("Machine Type", list(type_map.keys()))
machine_type = type_map[type_input]

selected_types = st.sidebar.multiselect(
    "Machine Type Filter",
    options=list(type_map.keys()),
    default=list(type_map.keys())
)
selected_failure = st.sidebar.multiselect(
    "Failure status",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Failure" if x == 1 else "No Failure"
)
selected_palette = st.sidebar.selectbox(
    "Chart color palette",
    options=["blues", "viridis", "matter", "plasma"],
    index=0
)
shap_sample_size = st.sidebar.slider(
    "SHAP sample size",
    min_value=50,
    max_value=min(500, len(df)),
    value=min(200, len(df)),
    step=50
)

# =========================
# INPUT FEATURE CREATION
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
    st.sidebar.warning("No data matches the filter. Showing full dataset.")
    filtered_df = df.copy()

filtered_df['Type_label'] = filtered_df['Type_label'].fillna('Unknown')
filtered_df['Failure_label'] = filtered_df['Machine failure'].map({0: 'No Failure', 1: 'Failure'})

predicted_proba = model.predict_proba(filtered_df[feature_cols])[:, 1] if len(filtered_df) > 0 else np.array([0])
filter_failure_rate = filtered_df['Machine failure'].mean() * 100 if len(filtered_df) > 0 else 0
avg_predicted_risk = predicted_proba.mean() * 100 if len(filtered_df) > 0 else 0

# =========================
# REAL-TIME PREDICTION
# =========================
st.subheader("🔮 Real-Time Downtime Prediction")

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("⚠ High Downtime Risk")
    else:
        st.success("✅ Low Downtime Risk")

with col2:
    st.metric("Failure Probability", f"{prob:.2f}")

prediction_tab, insights_tab, explain_tab = st.tabs([
    "Live Prediction",
    "Interactive Insights",
    "Explainability"
])

with prediction_tab:
    st.markdown("### Live Prediction Metrics")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Current Prediction", "Failure" if prediction == 1 else "No Failure")
    metric2.metric("Current Risk", f"{prob:.1f}%")
    metric3.metric("Risk Level", "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 44, 'color': 'white'}},
        title={'text': "Live Downtime Risk", 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': 'white'},
            'bar': {'color': '#00E676'},
            'steps': [
                {'range': [0, 30], 'color': '#0E7E34'},
                {'range': [30, 70], 'color': '#F5B400'},
                {'range': [70, 100], 'color': '#F35B04'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig_gauge.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin={'t': 10, 'b': 10, 'l': 10, 'r': 10}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# =========================
# SCHEDULING
# =========================
st.subheader("📅 Scheduling Recommendation")

if prediction == 1:
    st.error("🚨 Schedule Maintenance Immediately")
else:
    st.success("✅ Continue Production")

# =========================
# DATA INSIGHTS
# =========================
with insights_tab:
    st.subheader("📊 Interactive Data Insights")
    st.markdown("Use the filters on the left to explore the dataset in real time.")

    st.markdown("### Current Filters")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Machine Types:** {', '.join(selected_types) if selected_types else 'None'}")
    with col_b:
        st.write(f"**Failure Status:** {', '.join(['Failure' if x == 1 else 'No Failure' for x in selected_failure]) if selected_failure else 'None'}")

    st.info(f"Filtered dataset has {len(filtered_df)} records matching the current filters.")

    st.markdown("### Dataset Metrics")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Filtered records", len(filtered_df))
    metric2.metric("Filtered failure rate", f"{filter_failure_rate:.1f}%")
    metric3.metric("Avg dataset risk", f"{avg_predicted_risk:.1f}%")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig1 = px.histogram(
            filtered_df,
            x='Failure_label',
            color='Type_label',
            barmode='group',
            title="Failure Distribution by Machine Type",
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with chart_cols[1]:
        fig2 = px.box(
            filtered_df,
            x='Failure_label',
            y='Torque [Nm]',
            color='Failure_label',
            title="Torque by Failure Status",
            template='plotly_dark',
            color_discrete_sequence=['#00cc96', '#ff6361']
        )
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Custom scatter explorer")
    x_axis = st.selectbox("X axis feature", feature_cols, index=0)
    y_axis = st.selectbox("Y axis feature", feature_cols, index=3)
    fig_scatter = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color='Failure_label',
        title=f"{y_axis} vs {x_axis}",
        template='plotly_dark',
        hover_data=feature_cols,
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Feature Correlation")
    corr_df = filtered_df[feature_cols + ['Machine failure']].apply(pd.to_numeric, errors='coerce')
    corr = corr_df.corr()
    fig3 = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Matrix",
        template='plotly_dark',
        color_continuous_scale=selected_palette
    )
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

# Ensure numeric only (NO ERROR GUARANTEED)
corr_df = df[feature_cols + ['Machine failure']].apply(pd.to_numeric, errors='coerce')
corr = corr_df.corr()

fig3 = px.imshow(corr, text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# =========================
# SHAP
# =========================
with explain_tab:
    st.subheader("🧠 Explainable AI (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        X_sample = df[feature_cols].sample(min(shap_sample_size, len(df)))

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_plot_values = shap_values[1] if len(shap_values) == 2 else shap_values
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            shap_plot_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
        else:
            shap_plot_values = shap_values

        importance = np.mean(np.abs(shap_plot_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': importance
        }).sort_values('importance', ascending=True)

        fig_shap = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='SHAP Feature Importance',
            labels={'importance': 'Mean |SHAP value|', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale=selected_palette,
            template='plotly_dark'
        )
        fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as exc:
        st.warning(f"SHAP visualization could not be generated: {exc}")

# =========================
# SAMPLE PREDICTIONS
# =========================
st.subheader("🧪 Sample Predictions")

sample = df.sample(5).copy()
X_sample = sample[feature_cols]

sample['Prediction'] = model.predict(X_sample)

st.dataframe(sample)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Smart Manufacturing Downtime Prediction System (Real-Time)")
