import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

# Load models and scaler
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")

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
st.title("Power Output Prediction App ⚡")
st.markdown("Predict Total Power using ensemble of Random Forest and XGBoost.")

# Sidebar input
st.sidebar.header("Input Parameters")

def get_input():
    inputs = {}
    for feature in input_features:
        default = 25.0 if "Temperature" in feature else 50.0
        inputs[feature] = st.sidebar.number_input(f"{feature}", value=default)
    return inputs

user_input = get_input()

# Preprocessing
def preprocess(features_dict):
    values = list(features_dict.values())
    
    # Add interaction term
    temp = features_dict["Ambient Temperature"]
    humidity = features_dict["Ambient Relative Humidity"]
    interaction = temp * humidity
    values.append(interaction)

    # Apply same preprocessing as training
    values_array = np.array(values).reshape(1, -1)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(values_array)

    minmax_scaler = MinMaxScaler()
    scaled_features = minmax_scaler.fit_transform(poly_features)  # temp scaling before standard scaler
    final_features = standard_scaler.transform(scaled_features)
    
    return final_features

# Prediction
if st.button("Predict Power Output"):
    processed = preprocess(user_input)
    prediction = best_w * rf_model.predict(processed) + (1 - best_w) * xgb_model.predict(processed)
    st.success(f"🔋 Predicted Total Power Output: **{prediction[0]:.2f}** MW")

# Optional Info
st.markdown("---")
st.markdown("Model ensemble uses **{:.0f}% Random Forest** and **{:.0f}% XGBoost**.".format(best_w * 100, (1 - best_w) * 100))
