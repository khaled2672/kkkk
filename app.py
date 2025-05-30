import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set page config
st.set_page_config(page_title="Power Plant Optimization", layout="wide")

# Load models and scalers
@st.cache_resource
def load_models():
    feature_scaler = joblib.load("feature_scaler.pkl")
    standard_scaler = joblib.load("standard_scaler.pkl")
    rf_model = joblib.load("rf_model.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
    return feature_scaler, standard_scaler, rf_model, xgb_model, best_w

feature_scaler, standard_scaler, rf_model, xgb_model, best_w = load_models()

# Sidebar for user input
st.sidebar.header("Input Parameters")
st.sidebar.write("Adjust the parameters to predict power output:")

# Create input sliders
input_data = {}
col1, col2 = st.sidebar.columns(2)
with col1:
    input_data['Ambient Temperature'] = st.slider(
        'Ambient Temperature (°C)',
        min_value=0.0, max_value=50.0, value=25.0, step=0.1
    )
    input_data['Ambient Pressure'] = st.slider(
        'Ambient Pressure (mbar)',
        min_value=900.0, max_value=1100.0, value=1013.0, step=0.1
    )
with col2:
    input_data['Ambient Relative Humidity'] = st.slider(
        'Ambient Relative Humidity (%)',
        min_value=0.0, max_value=100.0, value=50.0, step=0.1
    )
    input_data['Exhaust Vacuum'] = st.slider(
        'Exhaust Vacuum (cmHg)',
        min_value=20.0, max_value=100.0, value=50.0, step=0.1
    )

# Main content
st.title("Power Plant Performance Optimizer")
st.write("""
This app predicts the total power output of a combined cycle power plant based on ambient conditions.
The model uses an ensemble of Random Forest and XGBoost algorithms.
""")

# Prepare input data
input_df = pd.DataFrame([input_data])
features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

# Scale the input data
scaled_input = feature_scaler.transform(input_df[features])
standard_scaled_input = standard_scaler.transform(scaled_input)

# Make predictions
rf_pred = rf_model.predict(standard_scaled_input)[0]
xgb_pred = xgb_model.predict(standard_scaled_input)[0]
combined_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# Display predictions
st.subheader("Power Output Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Random Forest Prediction", f"{rf_pred:.2f} MW")
col2.metric("XGBoost Prediction", f"{xgb_pred:.2f} MW")
col3.metric("Combined Prediction", f"{combined_pred:.2f} MW", delta=f"{(combined_pred - (rf_pred + xgb_pred)/2):.2f} vs average")

# Feature importance visualization
st.subheader("Feature Importance")
tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])

with tab1:
    try:
        importances = rf_model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importances, color='skyblue')
        ax.set_title('Random Forest Feature Importance')
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not display Random Forest feature importance")

with tab2:
    try:
        importances = xgb_model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importances, color='salmon')
        ax.set_title('XGBoost Feature Importance')
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not display XGBoost feature importance")

# Optimization section
st.subheader("Optimal Parameters for Maximum Power")
st.write("""
The system has found these optimal parameters that would maximize power output:
""")

# Display optimal parameters from PSO (you might want to pre-compute these)
optimal_scaled = np.array([0.5, 0.5, 0.5, 0.5])  # Replace with your actual PSO results
optimal_original = feature_scaler.inverse_transform(optimal_scaled.reshape(1, -1))[0]

optimal_params = dict(zip(features, optimal_original))
for param, value in optimal_params.items():
    st.write(f"- **{param}**: {value:.2f}")

# Comparison with current input
st.write("\n")
st.write("### Comparison with your input:")
comparison_df = pd.DataFrame({
    'Parameter': features,
    'Your Input': [input_data[f] for f in features],
    'Optimal Value': optimal_original,
    'Difference': [input_data[f] - optimal_original[i] for i, f in enumerate(features)]
})
st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)

# Footer
st.markdown("---")
st.write("""
**Note**: This is a predictive model and actual power plant performance may vary based on other factors not considered here.
""")
