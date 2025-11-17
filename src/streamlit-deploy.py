import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.markdown("Enter the details of the house to get an estimated price.")

# Load the model
@st.cache_resource
def load_model():
    model_path = "src/models/pipeline.bin"
    if not os.path.exists(model_path):
        st.write(f"Current working directory: {os.getcwd()}")
        st.error(f"Model file not found at {model_path}")
        st.stop()
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

# -----------------------------
# User Input Form
# -----------------------------
with st.form("house_form"):
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price (for reference only)", min_value=0, value=0, disabled=True)
        area = st.number_input("Area (sq ft)", min_value=0, value=5000, step=10)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
        parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1)

    with col2:
        mainroad = st.selectbox("Main road", options=["yes", "no"], index=0)
        guestroom = st.selectbox("Guest room", options=["yes", "no"], index=1)
        basement = st.selectbox("Basement", options=["yes", "no"], index=1)
        hotwaterheating = st.selectbox("Hot water heating", options=["yes", "no"], index=1)
        airconditioning = st.selectbox("Air conditioning", options=["yes", "no"], index=0)
        prefarea = st.selectbox("Preferred area", options=["yes", "no"], index=1)
        furnishingstatus = st.selectbox("Furnishing status", 
                                        options=["furnished", "semi-furnished", "unfurnished"], 
                                        index=1)

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Create input DataFrame exactly matching training columns
        input_data = {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "hotwaterheating": [hotwaterheating],
            "airconditioning": [airconditioning],
            "parking": [parking],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishingstatus]
        }
        
        st.write(input_data)
        # Prediction
        try:
            prediction = np.expm1(pipeline.predict(input_data)[0])
            prediction = float(prediction)
            st.write(prediction)
            st.success(f"### Predicted House Price: **${prediction:,.2f}**")

            # Optional: Show input summary
            with st.expander("View input details"):
                st.write(df.T.rename(columns={0: "Value"}))

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Make sure your pipeline.bin is compatible and was trained on these exact columns.")
