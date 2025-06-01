import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models and scaler
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Features including interaction
selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum', 'Temp*Humidity']

st.title("Power Prediction & Optimization App")

st.sidebar.header("Input Features")

# User inputs for features
ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 20.0, 30.0, 25.0)
ambient_humidity = st.sidebar.slider('Ambient Relative Humidity (%)', 40.0, 70.0, 55.0)
ambient_pressure = st.sidebar.slider('Ambient Pressure (kPa)', 799.0, 800.0, 799.5)
exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum', 4.5, 6.0, 5.0)

# Ensemble weight slider
weight = st.sidebar.slider('Ensemble Weight (Random Forest)', 0.0, 1.0, 0.5)

# Prepare input data with interaction feature
input_data = {
    'Ambient Temperature': ambient_temp,
    'Ambient Relative Humidity': ambient_humidity,
    'Ambient Pressure': ambient_pressure,
    'Exhaust Vacuum': exhaust_vacuum,
}

# Add interaction term
input_data['Temp*Humidity'] = input_data['Ambient Temperature'] * input_data['Ambient Relative Humidity']

# Create DataFrame
input_df = pd.DataFrame([input_data], columns=selected_features)

# Scale features
scaled_input = scaler.transform(input_df)

# Predict with RF and XGB
rf_pred = rf_model.predict(scaled_input)[0]
xgb_pred = xgb_model.predict(scaled_input)[0]

# Ensemble prediction
ensemble_pred = weight * rf_pred + (1 - weight) * xgb_pred

st.subheader("Predictions")
st.write(f"Random Forest Prediction: {rf_pred:.2f} MW")
st.write(f"XGBoost Prediction: {xgb_pred:.2f} MW")
st.write(f"Ensemble Prediction: {ensemble_pred:.2f} MW (Weight RF={weight:.2f}, XGB={1-weight:.2f})")

# Optional: show feature importance plots
if st.checkbox("Show Feature Importances"):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pd.Series(rf_model.feature_importances_, index=selected_features).plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title("Random Forest Feature Importance")
    pd.Series(xgb_model.feature_importances_, index=selected_features).plot(kind='bar', ax=ax2, color='salmon')
    ax2.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

# Optional: add a PSO optimizer button here (if you want)

