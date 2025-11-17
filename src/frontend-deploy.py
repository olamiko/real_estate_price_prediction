import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="house",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        border-left: 6px solid #3B82F6;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E40AF;
        margin: 20px 0;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3.2em;
        width: 100%;
        font-size: 1.1rem;
    }
    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #94A3B8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<h1 class="main-title">House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get accurate price estimates in seconds using machine learning</p>', unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    model_path = "src/models/pipeline.bin"
    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`")
        st.write(f"Current directory: `{os.getcwd()}`")
        st.stop()
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

with st.spinner("Initializing model..."):
    pipeline = load_model()

st.success("Model loaded successfully!")

# -----------------------------
# Input Form
# -----------------------------
st.markdown("### Enter House Details")

with st.form("prediction_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Area (sq ft)", min_value=1000, max_value=30000, value=5500, step=50,
                        help="Total land area in square feet")
        st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
        st.number_input("Stories", min_value=1, max_value=4, value=2)
        st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

    with col2:
        st.selectbox("Connected to Main Road", options=["yes", "no"], index=0)
        st.selectbox("Guest Room", options=["no", "yes"], index=0)
        st.selectbox("Basement", options=["no", "yes"], index=0)
        st.selectbox("Hot Water Heating", options=["no", "yes"], index=0)
        st.selectbox("Air Conditioning", options=["yes", "no"], index=0)
        st.selectbox("Preferred / Prime Area", options=["no", "yes"], index=0)
        st.selectbox("Furnishing Status", 
                     options=["unfurnished", "semi-furnished", "furnished"],
                     index=1)

    # Submit button
    submitted = st.form_submit_button("Predict House Price", use_container_width=True)

    if submitted:
        # Collect all inputs properly
        input_dict = {
            "area": col1.number_input("Area (sq ft)"),
            "bedrooms": col1.number_input("Bedrooms"),
            "bathrooms": col1.number_input("Bathrooms"),
            "stories": col1.number_input("Stories"),
            "mainroad": col2.selectbox("Connected to Main Road").lower(),
            "guestroom": col2.selectbox("Guest Room").lower(),
            "basement": col2.selectbox("Basement").lower(),
            "hotwaterheating": col2.selectbox("Hot Water Heating").lower(),
            "airconditioning": col2.selectbox("Air Conditioning").lower(),
            "parking": col1.number_input("Parking Spaces"),
            "prefarea": col2.selectbox("Preferred / Prime Area").lower(),
            "furnishingstatus": col2.selectbox("Furnishing Status")
        }

        input_df = pd.DataFrame([input_dict])

        with st.spinner("Predicting price..."):
            try:
                # Important: Use np.expm1 if target was log-transformed during training
                log_pred = pipeline.predict(input_df)[0]
                predicted_price = np.expm1(log_pred)
                predicted_price = float(predicted_price)

                # Beautiful result display
                st.markdown(f"""
                <div class="prediction-result">
                    ${predicted_price:,.0f}
                </div>
                """, unsafe_allow_html=True)

                st.balloons()

                # Show input summary
                with st.expander("View entered details", expanded=False):
                    display_df = input_df.copy()
                    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
                    st.dataframe(display_df.T.rename(columns={0: "Your Input"}), use_container_width=True)

            except Exception as e:
                st.error("Prediction failed. Check model compatibility.")
                st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Real Estate Price Prediction • Powered by Machine Learning • Model: src/models/pipeline.bin
</div>
""", unsafe_allow_html=True)
