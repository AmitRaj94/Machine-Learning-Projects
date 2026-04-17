import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Downtime Risk Predictor", layout="wide")

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===== TITLE =====
st.title("🏭 Smart Manufacturing Downtime Risk Dashboard")
st.markdown("Predict machine downtime risk and optimize production decisions.")

# ===== SIDEBAR INPUT =====
st.sidebar.header("🔧 Machine Inputs")

type_val = st.sidebar.selectbox("Machine Type", [0, 1, 2])

air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0)
rpm = st.sidebar.slider("RPM", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear", 0, 250, 100)

# ===== FEATURE ENGINEERING =====
temp_diff = process_temp - air_temp
power = rpm * torque
wear_per_rpm = tool_wear / rpm if rpm != 0 else 0

Type_1 = 1 if type_val == 1 else 0
Type_2 = 1 if type_val == 2 else 0

features = np.array([[
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    temp_diff,
    power,
    wear_per_rpm,
    Type_1,
    Type_2
]])

# ===== SCALE + PREDICT =====
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

# ===== MAIN DASHBOARD =====
col1, col2 = st.columns(2)

# ---- RESULT DISPLAY ----
with col1:
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Downtime Risk")
        risk_score = 80
    else:
        st.success("✅ Low Downtime Risk")
        risk_score = 20

    st.progress(risk_score)

# ---- INPUT SUMMARY ----
with col2:
    st.subheader("📋 Input Summary")
    st.write(f"Type: {type_val}")
    st.write(f"Air Temp: {air_temp}")
    st.write(f"Process Temp: {process_temp}")
    st.write(f"RPM: {rpm}")
    st.write(f"Torque: {torque}")
    st.write(f"Tool Wear: {tool_wear}")

# ===== VISUALIZATION =====
st.subheader("📈 Operational Insights")

data = pd.DataFrame({
    "Feature": ["RPM", "Torque", "Tool Wear"],
    "Value": [rpm, torque, tool_wear]
})

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(data["Feature"], data["Value"])
st.pyplot(fig,use_container_width=False)

# ===== SMART RECOMMENDATIONS =====
st.subheader("🧠 AI Recommendation")

if prediction == 1:
    st.warning("Reduce load, inspect tool wear, or schedule maintenance.")
else:
    st.info("Machine operating within safe parameters.")

# ===== REAL-TIME FEEDBACK =====
st.subheader("⚡ Live Risk Indicators")

if tool_wear > 150:
    st.write("🔴 High Tool Wear Detected")

if torque > 50:
    st.write("🟠 High Torque Load")

if rpm > 2500:
    st.write("🟡 High RPM Stress")

# ===== FOOTER =====
st.markdown("---")
st.caption("AI-powered predictive maintenance system")
