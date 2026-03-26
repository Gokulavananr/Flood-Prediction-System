import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Flood Prediction", layout="wide")

st.title("🌊 Flood Impact Prediction System")
st.markdown("### Advanced Disaster Prediction Dashboard")

# 🔹 Sidebar Inputs
st.sidebar.header("Enter Input Values")

latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 13.0)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, 80.0)
total_deaths = st.sidebar.slider("Total Deaths", 0, 500, 10)
total_affected = st.sidebar.slider("Total Affected", 0, 50000, 1000)
duration = st.sidebar.slider("Duration (days)", 0, 30, 5)
time = st.sidebar.slider("Time", 0, 24, 12)
rainfall = st.sidebar.slider("Rainfall", 0, 500, 100)
elevation = st.sidebar.slider("Elevation", 0, 500, 50)
slope = st.sidebar.slider("Slope", 0, 20, 5)
distance = st.sidebar.slider("Distance", 0, 50, 10)

# Input array
input_data = np.array([[latitude, longitude, total_deaths, total_affected,
                        duration, time, rainfall, elevation, slope, distance]])

# 🔹 Prediction Button
if st.button("🚀 Predict Now"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # 🔥 Risk Level Logic
    if probability > 0.7:
        risk = "🔴 HIGH RISK"
    elif probability > 0.4:
        risk = "🟠 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    # 🔹 Result Display
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ Disaster Likely ({risk})")
        else:
            st.success(f"✅ No Disaster Expected ({risk})")

        st.metric("Prediction Probability", f"{probability*100:.2f}%")

    # 🔹 Graph Visualization
    with col2:
        st.subheader("📊 Feature Analysis")

        features = ["Rainfall", "Elevation", "Slope", "Distance"]
        values = [rainfall, elevation, slope, distance]

        fig = px.bar(x=features, y=values, title="Environmental Factors")
        st.plotly_chart(fig)

    # 🔹 Map Visualization
    st.subheader("🗺️ Location Map")

    map_data = pd.DataFrame({
        "lat": [latitude],
        "lon": [longitude]
    })

    st.map(map_data)

    # 🔹 Input Summary Table
    st.subheader("📋 Input Summary")

    df = pd.DataFrame(input_data, columns=[
        "Latitude", "Longitude", "Deaths", "Affected",
        "Duration", "Time", "Rainfall", "Elevation", "Slope", "Distance"
    ])

    st.dataframe(df)

    # 🔹 Download Report
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download Input Data", csv, "report.csv")