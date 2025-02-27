import streamlit as st
import pickle
import numpy as np
import mysql.connector
import datetime
from database import fetch_latest_data

# Load the trained model
with open("fare_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🚖 Optimized Uber Ride Booking")
st.subheader("Find the best time to book an Uber at the lowest fare")

# User Inputs
pickup_location = st.text_input("📍 Enter Pickup Location:")
destination = st.text_input("📍 Enter Destination:")
pickup_time = st.time_input("⏰ Select Pickup Time:")

if st.button("🔍 Get Fare Prediction"):
    # Convert pickup time to a numeric feature (hour of the day)
    time_numeric = int(pickup_time.strftime('%H'))

    # Fetch real-time data (if available)
    latest_data = fetch_latest_data()
    
    # Predict fare
    predicted_fare = model.predict(np.array([[time_numeric]]))[0]

    # Display results
    st.success(f"Predicted fare for ride from **{pickup_location}** to **{destination}** at **{pickup_time}**: **${predicted_fare:.2f}**")

    if latest_data:
        st.info(f"Current estimated fare: **${latest_data['fare']}** | Wait Time: **{latest_data['wait_time']} mins**")
