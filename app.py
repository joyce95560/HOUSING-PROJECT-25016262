import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load your Model and the 80-feature template
# Make sure 'feature_template.csv' has all 80 columns (including Id)
try:
    rf_model = joblib.load('best_house_predictor.pkl')
    lr_model = joblib.load('house_model.pkl')
    template = pd.read_csv('feature_template.csv')
except Exception as e:
    st.error(f"Error loading files: {e}")

st.title("🏠 House Price Predictor")

# 2. User Inputs (The "Big 6")
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area (sq ft)", value=1500)
    year_built = st.number_input("Year Built", 1870, 2024, 2005)
with col2:
    garage_cars = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", value=1000)

model_choice = st.radio("Select Model", ("Random Forest", "Linear Regression"))

# 3. PREDICTION LOGIC
if st.button("Predict Sale Price"):
    # STEP A: Create the 80-column row from your template
    input_data = template.copy()

    # STEP B: Fix the 'Id' Error (Now it's defined inside the button click!)
    # We give it a dummy ID because the model expects one
    if 'Id' in input_data.columns:
        input_data['Id'] = 0 

    # STEP C: Update the features the user actually changed
    # (Ensure these names match your train.csv exactly)
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['GarageCars'] = garage_cars
    input_data['FullBath'] = full_bath
    input_data['TotalBsmtSF'] = total_bsmt_sf

    # STEP D: Final Prediction
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
    else:
        prediction = lr_model.predict(input_data)

    st.success(f"### Predicted Sale Price: ${prediction[0]:,.2f}")
    st.balloons()
