import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# =============================
# Page Config & Beautiful Styling
# =============================
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="house",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a premium feel
st.markdown("""
<style>
    .big-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1E40AF, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 8px solid #3b82f6;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #1e40af;
        margin: 30px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        height: 3.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown('<h1 class="big-title">House Price Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter property details below to get an instant AI-powered price estimate</p>', unsafe_allow_html=True)

# =============================
# Load Model
# =============================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    model_path = "src/models/pipeline.bin"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.write(f"Expected: `{model_path}`")
        st.write(f"Current directory: `{os.getcwd()}`")
        st.stop()
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()
st.success("Model loaded successfully!")

# =============================
# User Input Form
# =============================
with st.form("house_form", clear_on_submit=True):
    st.markdown("### Property Features")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Area (sq ft)", min_value=1000, max_value=50000, value=5500, step=100, help="Total built-up area")
        st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
        st.number_input("Stories", min_value=1, max_value=4, value=2)
        st.number_input("Parking spaces", min_value=0, max_value=5, value=1)

    with col2:
        mainroad = st.selectbox("Main road access", options=["yes", "no"], index=0)
        guestroom = st.selectbox("Guest room", options=["no", "yes"], index=0)
        basement = st.selectbox("Basement", options=["no", "yes"], index=0)
        hotwaterheating = st.selectbox("Hot water heating", options=["no", "yes"], index=0)
        airconditioning = st.selectbox("Air conditioning", options=["yes", "no"], index=0)
        prefarea = st.selectbox("Preferred area", options=["no", "yes"], index=0)
        furnishingstatus = st.selectbox(
            "Furnishing status",
            options=["unfurnished", "semi-furnished", "furnished"],
            index=1
        )

    # Predict Button
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        # Build input dictionary using the captured variables
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }

        # Convert to DataFrame (single row)
        input_df = pd.DataFrame([input_data])

        with st.spinner("Estimating price..."):
            try:
                # Prediction with log transform reversal
                predicted_price = float(np.expm1(pipeline.predict(input_df)[0]))

                # Beautiful result
                st.markdown(f"""
                <div class="prediction-box">
                    ${predicted_price:,.0f}
                </div>
                """, unsafe_allow_html=True)

                st.balloons()

                # Input summary
                with st.expander("Show entered details", expanded=False):
                    summary = pd.DataFrame.from_dict(input_data, orient='index', columns=['Value'])
                    summary.index = summary.index.str.replace('_', ' ').str.title()
                    st.dataframe(summary, use_container_width=True)

            except Exception as e:
                st.error("Prediction failed")
                st.caption("Check that your model expects the same column names and 'yes'/'no' string format.")
                st.exception(e)

# =============================
# Footer
# =============================
st.markdown(
    "<p style='text-align: center; color: #94a3b8; margin-top: 60px; font-size: 0.9rem;'>"
    "Real Estate Price Prediction • Powered by Machine Learning • 2025"
    "</p>",
    unsafe_allow_html=True
)
