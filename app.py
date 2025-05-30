import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load models and preprocessing objects
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
poly = joblib.load("poly_transform.pkl")
minmax_scaler = joblib.load("minmax_scaler.pkl")

with open("best_weight.txt", "r") as f:
    best_w = float(f.read().strip())

# Define input features
input_features = [
    "Ambient Temperature",
    "Ambient Relative Humidity",
    "Ambient Pressure",
    "Exhaust Vacuum"
]

# Streamlit UI
st.title("⚡ Power Output Prediction App")
st.markdown("Predict **Total Power** using an ensemble of Random Forest and XGBoost.")

# Sidebar input
st.sidebar.header("Input Parameters")

def get_input():
    inputs = {}
    for feature in input_features:
        default = 25.0 if "Temperature" in feature else 50.0
        inputs[feature] = st.sidebar.number_input(f"{feature}", value=default)
    return inputs

user_input = get_input()

# Preprocessing function
def preprocess(features_dict):
    values = list(features_dict.values())
    temp = features_dict["Ambient Temperature"]
    humidity = features_dict["Ambient Relative Humidity"]
    interaction = temp * humidity
    values.append(interaction)

    # Convert to array and apply saved transformers
    values_array = np.array(values).reshape(1, -1)
    poly_features = poly.transform(values_array)
    scaled_features = minmax_scaler.transform(poly_features)
    final_features = standard_scaler.transform(scaled_features)
    return final_features

# Predict
if st.button("Predict Power Output"):
    processed = preprocess(user_input)
    prediction = best_w * rf_model.predict(processed) + (1 - best_w) * xgb_model.predict(processed)
    st.success(f"🔋 Predicted Total Power Output: **{prediction[0]:.2f}** MW")

st.markdown("---")
st.markdown(f"Model ensemble uses **{best_w * 100:.0f}% Random Forest** and **{(1 - best_w) * 100:.0f}% XGBoost**.")
