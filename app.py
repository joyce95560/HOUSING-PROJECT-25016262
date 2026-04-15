import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD MODELS AND TEMPLATE
# These must be uploaded to your GitHub repository
try:
    rf_model = joblib.load('best_house_predictor.pkl')
    lr_model = joblib.load('house_model.pkl')
    template = pd.read_csv('feature_template.csv')
except Exception as e:
    st.error(f"Error loading files: {e}. Make sure .pkl and .csv files are in GitHub.")

# 2. PAGE INTERFACE
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 Advanced House Price Predictor")
st.write("Enter the key features below. The model will handle the other 73 technical details automatically.")

# 3. USER INPUTS (The Big 6)
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
    year_built = st.number_input("Year Built", 1870, 2024, 2005)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", value=1000)

model_choice = st.sidebar.radio("Select Prediction Model", ("Random Forest", "Linear Regression"))

# 4. PREDICTION LOGIC
if st.button("Predict Sale Price"):
    input_data['Id'] = 0

    # Update ONLY the 6 columns we have user inputs for
    # Make sure these names match your train.csv exactly
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['GarageCars'] = garage_cars
    input_data['FullBath'] = full_bath
    input_data['TotalBsmtSF'] = total_bsmt_sf

    # Run the prediction
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
        st.success(f"### Random Forest Estimate: ${prediction[0]:,.2f}")
    else:
        prediction = lr_model.predict(input_data)
        st.success(f"### Linear Regression Estimate: ${prediction[0]:,.2f}")
    
    st.balloons()

st.info("This app uses a 'Feature Template' to prevent errors by filling in missing technical data with dataset medians.")
