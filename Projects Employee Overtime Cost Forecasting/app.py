import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
model = joblib.load("overtime_cost_model.pkl")

st.set_page_config(page_title="Overtime Forecast", layout="wide")

st.title("💼 Employee Overtime Cost Forecasting Dashboard")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📥 Input Workforce Data")

employee_tenure_months = st.sidebar.slider("months", 1, 240, 60)
base_hourly_rate_usd = st.sidebar.number_input("Hourly Rate (₹)", 10.0, 1000.0, 100.0)
scheduled_hours = st.sidebar.slider("Scheduled Hours", 20, 60, 40)
overtime_hours = st.sidebar.slider("Overtime Hours", 0.0, 20.0, 5.0)
ot_rate_multiplier = st.sidebar.selectbox("OT Multiplier", [1.5, 1.75, 2.0])

team_size = st.sidebar.slider("Team Size", 5, 60, 20)
absence_rate = st.sidebar.slider("Absence Rate", 0.0, 0.5, 0.1)
workload_index = st.sidebar.slider("Workload Index", 1.0, 10.0, 5.0)
prior_week_overtime_hours = st.sidebar.slider("Last Week OT", 0.0, 20.0, 5.0)

day_of_week = st.sidebar.selectbox("Day (0=Mon)", list(range(7)))
week_of_year = st.sidebar.slider("Week", 1, 52, 25)
month = st.sidebar.slider("Month", 1, 12, 6)
quarter = st.sidebar.selectbox("Quarter", [1,2,3,4])
year = st.sidebar.slider("Year", 2022, 2036, 2028)
day = st.sidebar.slider("Day", 1, 31, 15)

is_weekend = st.sidebar.selectbox("Weekend?", [0,1])
is_holiday_week = st.sidebar.selectbox("Holiday Week?", [0,1])
is_peak_season = st.sidebar.selectbox("Peak Season?", [0,1])

dept = st.sidebar.selectbox("Department", 
    ["IT Support", "Logistics", "Manufacturing", "Security", "Warehouse"])

location = st.sidebar.selectbox("Location", 
    ["HQ", "Plant A", "Plant B", "Plant C"])

shift = st.sidebar.selectbox("Shift", 
    ["Night", "Rotating", "Split"])

contract = st.sidebar.selectbox("Contract", 
    ["Full-time", "Part-time"])

# =========================
# BUILD INPUT DATA
# =========================
input_dict = {
    "employee_tenure_months": employee_tenure_months,
    "base_hourly_rate_usd": base_hourly_rate_usd,
    "scheduled_hours": scheduled_hours,
    "overtime_hours": overtime_hours,
    "ot_rate_multiplier": ot_rate_multiplier,
    "day_of_week": day_of_week,
    "week_of_year": week_of_year,
    "month": month,
    "quarter": quarter,
    "year": year,
    "is_weekend": is_weekend,
    "is_holiday_week": is_holiday_week,
    "is_peak_season": is_peak_season,
    "team_size": team_size,
    "absence_rate": absence_rate,
    "workload_index": workload_index,
    "prior_week_overtime_hours": prior_week_overtime_hours,
    "day": day
}

input_df = pd.DataFrame([input_dict])

# One-hot columns
all_cols = model.feature_names_in_

for col in all_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df[f"department_{dept}"] = 1
input_df[f"location_{location}"] = 1
input_df[f"shift_type_{shift}"] = 1
input_df[f"contract_type_{contract}"] = 1

input_df = input_df[all_cols]

# =========================
# PREDICTION (REAL-TIME)
# =========================
prediction = model.predict(input_df)[0]

# =========================
# DISPLAY OUTPUT
# =========================
st.subheader("💰 Predicted Overtime Cost")

st.metric("Estimated Cost (₹)", f"{prediction:.2f}")

# =========================
# INPUT SUMMARY
# =========================
st.subheader("📋 Input Summary")
st.dataframe(input_df.T)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Top Feature Importance")

importance = pd.Series(model.feature_importances_, index=all_cols)
top_features = importance.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_features.plot(kind="barh", ax=ax)
ax.invert_yaxis()
st.pyplot(fig)

# =========================
# BUSINESS INSIGHTS
# =========================
st.subheader("🧠 Insights")

if overtime_hours > 8:
    st.warning("High overtime hours → increasing cost")

if absence_rate > 0.2:
    st.warning("High absence rate → workforce shortage")

if workload_index > 7:
    st.warning("High workload → more overtime demand")

if team_size < 10:
    st.warning("Small team size → higher overtime pressure")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed for HR Overtime Forecasting System")