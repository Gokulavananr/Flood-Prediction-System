import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from geopy.geocoders import Nominatim
import random

# Load models
model = pickle.load(open("model.pkl", "rb"))
death_model = pickle.load(open("death_model.pkl", "rb"))

# Geolocator
geolocator = Nominatim(user_agent="disaster_app")

st.set_page_config(page_title="Flood Prediction", layout="wide")

st.title("🌊 Flood Impact Prediction System")
st.markdown("### Advanced Disaster Prediction Dashboard")

# 🔹 Location Input
st.subheader("📍 Enter Location")

location_name = st.text_input("Enter Location (e.g., Chennai)")

# Default values
latitude = 13.0
longitude = 80.0

# Default environmental values
rainfall = 100
elevation = 50
slope = 5
distance = 10

# 🔹 Location Detection
if location_name:
    location = geolocator.geocode(location_name)
    
    if location:
        latitude = location.latitude
        longitude = location.longitude
        
        st.success(f"📍 Location Found: {latitude:.4f}, {longitude:.4f}")

        # 🔥 AUTO GENERATE DATA
        rainfall = random.randint(50, 300)
        elevation = random.randint(5, 100)
        slope = random.randint(1, 10)
        distance = random.randint(1, 20)

        st.info(f"Auto Data → Rainfall: {rainfall}, Elevation: {elevation}, Slope: {slope}, Distance: {distance}")

    else:
        st.error("❌ Location not found. Using default values.")

# 🔹 Sidebar Inputs
st.sidebar.header("Enter Input Values")

latitude = st.sidebar.slider("Latitude", -90.0, 90.0, float(latitude))
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, float(longitude))

total_deaths = st.sidebar.slider("Total Deaths", 0, 500, 10)
total_affected = st.sidebar.slider("Total Affected", 0, 50000, 1000)
duration = st.sidebar.slider("Duration (days)", 0, 30, 5)
time = st.sidebar.slider("Time", 0, 24, 12)

# Auto values in sliders
rainfall = st.sidebar.slider("Rainfall", 0, 500, int(rainfall))
elevation = st.sidebar.slider("Elevation", 0, 500, int(elevation))
slope = st.sidebar.slider("Slope", 0, 20, int(slope))
distance = st.sidebar.slider("Distance", 0, 50, int(distance))

# 🔹 Input array
input_data = np.array([[latitude, longitude, total_deaths, total_affected,
                        duration, time, rainfall, elevation, slope, distance]])

# 🔹 Prediction
if st.button("🚀 Predict Now"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Death prediction
    predicted_deaths = death_model.predict(input_data)[0]

    # Risk level
    if probability > 0.7:
        risk = "🔴 HIGH RISK"
    elif probability > 0.4:
        risk = "🟠 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ Disaster Likely ({risk})")
        else:
            st.success(f"✅ No Disaster Expected ({risk})")

        st.metric("Prediction Probability", f"{probability*100:.2f}%")
        st.warning(f"⚠️ Estimated Deaths: {int(predicted_deaths)}")

    with col2:
        st.subheader("📊 Feature Analysis")

        features = ["Rainfall", "Elevation", "Slope", "Distance"]
        values = [rainfall, elevation, slope, distance]

        fig = px.bar(x=features, y=values, title="Environmental Factors")
        st.plotly_chart(fig)

    # Map
    st.subheader("🗺️ Location Map")

    map_data = pd.DataFrame({
        "lat": [latitude],
        "lon": [longitude]
    })

    st.map(map_data)

    # Table
    st.subheader("📋 Input Summary")

    df = pd.DataFrame(input_data, columns=[
        "Latitude", "Longitude", "Deaths", "Affected",
        "Duration", "Time", "Rainfall", "Elevation", "Slope", "Distance"
    ])

    st.dataframe(df)

    # Download
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download Input Data", csv, "report.csv")


# 🤖 CHATBOT (ADDED AT BOTTOM)
st.subheader("🤖 AI Disaster Assistant")

user_query = st.text_input("Ask something (e.g., Flood risk in Chennai)")

if user_query:
    query = user_query.lower()

    if "chennai" in query:
        st.write("🌊 Chennai has high flood risk due to heavy rainfall and coastal location.")
    elif "mumbai" in query:
        st.write("🌧️ Mumbai is prone to flooding due to monsoon rains and drainage issues.")
    elif "delhi" in query:
        st.write("🌫️ Delhi has lower flood risk but can face urban flooding during heavy rains.")
    else:
        st.write("⚠️ Please enter a valid city like Chennai, Mumbai, or Delhi.")