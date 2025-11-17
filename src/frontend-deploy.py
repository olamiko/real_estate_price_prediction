import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime
import numpy as np

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="house",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1E3A8A; text-align: center; margin-bottom: 10px;}
    .subtitle {text-align: center; color: #64748B; margin-bottom: 30px;}
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(90deg, #E0F2FE, #CCEAFD);
        border-left: 6px solid #0EA5E9;
        margin: 20px 0;
    }
    .stButton>button {
        background-color: #0EA5E9;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .footer {text-align: center; margin-top: 50px; color: #94A3B8; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource(show_spinner="Loading prediction model...")
def load_pipeline():
    model_path = "src/models/pipeline.bin"
    if not os.path.exists(model_path):
        st.error("Model file `models/pipeline.bin` not found. Please check the path.")
        st.stop()
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    st.success("Model loaded successfully!")
    return pipeline

pipeline = load_pipeline()

# -----------------------------
# Title & Header
# -----------------------------
st.markdown('<h1 class="main-header">House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get instant house price estimates using AI</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar Info
# -----------------------------
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a trained machine learning model to predict house prices 
    based on location, size, and features.
    """)
    
    st.header("Features Used")
    st.write("""
    - Area, bedrooms, bathrooms
    - Stories, parking
    - Main road, guestroom, basement
    - AC, hot water, pref. area
    - Furnishing status
    """)
    
    st.caption(f"Model loaded: `pipeline.bin` • {datetime.now().strftime('%b %d, %Y')}")

# -----------------------------
# Tabs: Single Prediction | Batch Upload
# -----------------------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])
# tab1 = st.tabs(["Single Prediction"])

# ================================
# TAB 1: Single Prediction
# ================================
with tab1:
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Enter House Details")

        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Area (sq ft)", min_value=1650, max_value=20000, value=5000, step=100, help="Total plot area")
            st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
            st.selectbox("Main Road", options=["yes", "no"], index=0, help="Is it connected to main road?")
            
        with col2:
            st.number_input("Stories", min_value=1, max_value=4, value=2)
            st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
            st.selectbox("Air Conditioning", options=["yes", "no"], index=0)
            st.selectbox("Preferred Area", options=["yes", "no"], index=1, help="Prime/preferred locality")

        col3, col4 = st.columns(2)
        with col3:
            st.selectbox("Guest Room", options=["no", "yes"], index=0)
            st.selectbox("Basement", options=["no", "yes"], index=0)
        
        with col4:
            st.selectbox("Hot Water Heating", options=["no", "yes"], index=0)
            st.selectbox("Furnishing Status", 
                        options=["semi-furnished", "unfurnished", "furnished"],
                        index=0,
                        help="furnished > semi-furnished > unfurnished")

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

        if submitted:
            # Prepare input
            # data = {
            #     "area": [st.session_state.get("area", 5000)] if "area" not in st.session_state else [st.form_data.area],
            #     "bedrooms": [col1.number_input("Bedrooms").value],
            #     "bathrooms": [col1.number_input("Bathrooms").value],
            #     "stories": [col2.number_input("Stories").value],
            #     "mainroad": [col1.selectbox("Main Road").lower()],
            #     "guestroom": [col3.selectbox("Guest Room").lower()],
            #     "basement": [col4.selectbox("Basement").lower()],
            #     "hotwaterheating": [col4.selectbox("Hot Water Heating").lower()],
            #     "airconditioning": [col2.selectbox("Air Conditioning").lower()],
            #     "parking": [col2.number_input("Parking Spaces").value],
            #     "prefarea": [col2.selectbox("Preferred Area").lower()],
            #     "furnishingstatus": [col4.selectbox("Furnishing Status")]
            # }
            
            # Better way: use form context
            # input_df = pd.DataFrame({
            #     "area": [col1.number_input("Area (sq ft)")],
            #     "bedrooms": [col1.number_input("Bedrooms")],
            #     "bathrooms": [col1.number_input("Bathrooms")],
            #     "stories": [col2.number_input("Stories")],
            #     "mainroad": [col1.selectbox("Main Road")],
            #     "guestroom": [col3.selectbox("Guest Room")],
            #     "basement": [col4.selectbox("Basement")],
            #     "hotwaterheating": [col4.selectbox("Hot Water Heating")],
            #     "airconditioning": [col2.selectbox("Air Conditioning")],
            #     "parking": [col2.number_input("Parking Spaces")],
            #     "prefarea": [col2.selectbox("Preferred Area")],
            #     "furnishingstatus": [col4.selectbox("Furnishing Status")]
            # }).infer_objects()
            
            input_data = {
                "area": col1.number_input("Area (sq ft)"),
                "bedrooms": col1.number_input("Bedrooms"),
                "bathrooms": col1.number_input("Bathrooms"),
                "stories": col2.number_input("Stories"),
                "mainroad": col1.selectbox("Main Road"),
                "guestroom": col3.selectbox("Guest Room"),
                "basement": col4.selectbox("Basement"),
                "hotwaterheating": col4.selectbox("Hot Water Heating"),
                "airconditioning": col2.selectbox("Air Conditioning"),
                "parking": col2.number_input("Parking Spaces"),
                "prefarea": col2.selectbox("Preferred Area"),
                "furnishingstatus": col4.selectbox("Furnishing Status")
            }

            with st.spinner("Predicting price..."):
                try:
                    pred = pipeline.predict(input_data)[0]
                    pred = float(pred)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="margin:0; color:#0F172A;">Predicted Price</h2>
                        <h1 style="margin:5px 0; color:#0EA5E9; font-size:3rem;">
                            ${pred:,.0f}
                        </h1>
                        <p style="margin:0; color:#475569;">± typical model variance</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Show input summary"):
                        # st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
                        st.write(input_data) 

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Check if your model expects the same column names and dtypes (e.g., 'yes'/'no' as strings).")

# # ================================
# # TAB 2: Batch Prediction
# # ================================
# with tab2:
#     st.info("Upload a CSV with the exact columns as in training (except 'price')")
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             expected_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
#                            'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
            
#             if list(df.columns) != expected_cols:
#                 st.error(f"Columns don't match. Expected: {expected_cols}")
#                 st.write("Your columns:", list(df.columns))
#             else:
#                 with st.spinner(f"Predicting prices for {len(df)} houses..."):
#                     predictions = pipeline.predict(df)
#                     df["predicted_price"] = predictions.astype(int)
                    
#                     st.success(f"Batch prediction complete!")
#                     st.dataframe(df.style.format({"predicted_price": "${:,.0f}"}))

#                     csv = df.to_csv(index=False).encode()
#                     st.download_button(
#                         "Download Predictions",
#                         data=csv,
#                         file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
#                         mime="text/csv"
#                     )
#         except Exception as e:
#             st.error(f"Error processing file: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Built with Streamlit • Model: pipeline.bin • Made for real estate insights
</div>
""", unsafe_allow_html=True)
