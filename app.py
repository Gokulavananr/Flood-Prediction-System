import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🌊 Flood Impact Prediction System")

st.write("Enter environmental details:")

# Inputs
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
total_deaths = st.number_input("Total Deaths")
total_affected = st.number_input("Total Affected")
duration = st.number_input("Duration")
time = st.number_input("Time")
rainfall = st.number_input("Rainfall")
elevation = st.number_input("Elevation")
slope = st.number_input("Slope")
distance = st.number_input("Distance")

if st.button("Predict"):
    input_data = np.array([[latitude, longitude, total_deaths, total_affected,
                            duration, time, rainfall, elevation, slope, distance]])

    prediction = model.predict(input_data)

    # Prediction result
    if prediction[0] == 1:
        st.error("⚠️ Disaster Likely to Occur")
    else:
        st.success("✅ No Disaster Expected")

    # 📊 GRAPH (Corrected)
    st.subheader("📊 Input Feature Visualization")

    values = [rainfall, elevation, slope, distance]
    labels = ["Rainfall", "Elevation", "Slope", "Distance"]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Values")

    st.pyplot(fig)