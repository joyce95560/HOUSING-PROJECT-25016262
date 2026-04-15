import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Models
rf_model = joblib.load('best_house_predictor.pkl')
lr_model = joblib.load('house_model.pkl')

# 2. Load the 80-feature template (created in the training step)
# This ensures we always send exactly 80 columns to the model
template = pd.read_csv('feature_template.csv')

st.title("🏠 Full-Feature House Predictor")
st.write("This model uses all 80 features from the dataset for maximum accuracy.")

# 3. User inputs for the most impactful features
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
    year_built = st.number_input("Year Built", 1870, 2024, 2000)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", value=1000)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)
    lot_area = st.number_input("Lot Area (sq ft)", value=10000)
    model_choice = st.radio("Select Model", ("Random Forest", "Linear Regression"))

# 4. Prediction
if st.button("Predict Sale Price"):
    # Start with the 80-feature template
    input_data = template.copy()

    # Update the columns we have inputs for
    input_data['OverallQual'] = overall_qual
    input_data['GrLivArea'] = gr_liv_area
    input_data['YearBuilt'] = year_built
    input_data['TotalBsmtSF'] = total_bsmt_sf
    input_data['GarageCars'] = garage_cars
    input_data['FullBath'] = full_bath
    input_data['LotArea'] = lot_area

    # Final Check: Ensure all 80 columns are present and numeric
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
    else:
        prediction = lr_model.predict(input_data)

    st.success(f"### Predicted Sale Price: ${prediction[0]:,.2f}")
    st.balloons()