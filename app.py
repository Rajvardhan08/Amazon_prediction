import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("delivery_time_pipeline.pkl")

model = load_model()

st.set_page_config(page_title="Amazon Delivery Time Prediction", layout="centered")

st.title("📦 Amazon Delivery Time Prediction")
st.write("Predict estimated delivery time based on order and delivery conditions.")

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("Enter Delivery Details")

agent_age = st.slider("Agent Age", 18, 60, 30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5)

distance_km = st.slider("Distance (km)", 1.0, 25.0, 7.0)

order_hour = st.slider("Order Hour", 0, 23, 18)
order_day = st.selectbox(
    "Order Day",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ][x]
)

pickup_delay = st.slider("Pickup Delay (minutes)", 0, 20, 10)

traffic = st.selectbox(
    "Traffic Condition",
    ["Low", "Medium", "High", "Jam"]
)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"]
)

vehicle = st.selectbox(
    "Vehicle Type",
    ["motorcycle", "scooter"]
)

area = st.selectbox(
    "Delivery Area",
    ["Urban", "Metropolitian", "Semi-Urban", "Other"]
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Delivery Time"):
    input_df = pd.DataFrame([{
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Distance_km': distance_km,
        'Order_Hour': order_hour,
        'Order_Day': order_day,
        'Pickup_Delay_Min': pickup_delay,
        'Traffic': traffic,
        'Weather': weather,
        'Vehicle': vehicle,
        'Area': area
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

    st.caption(
        "Note: Traffic impact is reflected indirectly through pickup delays "
        "and distance-based routing patterns learned from historical data."
    )
