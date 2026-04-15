import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD THE MODELS ---
# Loading the models using your specific filenames
try:
    # Assuming 'best_house_predictor' is your Random Forest
    rf_model = joblib.load('best_house_predictor.pkl')
    # Assuming 'house_model' is your Linear Regression
    lr_model = joblib.load('house_model.pkl')
except Exception as e:
    st.error("Error: Model files not found. Please ensure 'best_house_predictor.pkl' and 'house_model.pkl' are in your GitHub repository.")

# --- 2. PAGE INTERFACE ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 Real Estate Price Predictor")
st.write("Enter the house details below to estimate the Sale Price.")

# Sidebar for model choice
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ("Random Forest (Best House Predictor)", "Linear Regression (House Model)")
)

# --- 3. INPUT FIELDS ---
st.subheader("Property Features")
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area (sq ft)", value=1500)
    year_built = st.number_input("Year Built", 1870, 2024, 2005)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", value=1000)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Sale Price"):
    # Features must be in the exact order used during training
    features = np.array([[
        overall_qual, 
        gr_liv_area, 
        garage_cars, 
        total_bsmt_sf, 
        full_bath, 
        year_built
    ]])

    if model_choice == "Random Forest (Best House Predictor)":
        prediction = rf_model.predict(features)
        st.success(f"### Estimated Price (RF): ${prediction[0]:,.2f}")
    else:
        prediction = lr_model.predict(features)
        st.success(f"### Estimated Price (LR): ${prediction[0]:,.2f}")
    
    st.balloons()

st.info("This app uses models trained on the House Prices dataset for educational purposes.")