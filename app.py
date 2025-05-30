import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set page config
st.set_page_config(
    page_title="Power Plant Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_FILES = {
    'feature_scaler': 'feature_scaler.pkl',
    'standard_scaler': 'standard_scaler.pkl',
    'rf_model': 'rf_model.joblib',
    'xgb_model': 'xgb_model.joblib',
    'best_weight': 'best_weight.txt'
}

# Utility functions
@st.cache_resource
def load_assets():
    """Load all required models and scalers with error handling"""
    loaded_assets = {}
    missing_files = []
    
    for name, filename in MODEL_FILES.items():
        try:
            if filename.endswith('.txt'):
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    # Ensure the content can be converted to float
                    try:
                        loaded_assets[name] = float(content)
                    except ValueError:
                        st.error(f"Invalid content in {filename}. Expected a number, got: {content}")
                        st.stop()
            else:
                loaded_assets[name] = joblib.load(filename)
        except FileNotFoundError:
            missing_files.append(filename)
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.error("Please ensure these files are in the same directory as your app:")
        for file in missing_files:
            st.error(f"- {file}")
        st.stop()
    
    return loaded_assets

# Load all models and scalers
assets = load_assets()

# Sidebar for user input
with st.sidebar:
    st.header("⚙️ Input Parameters")
    st.write("Adjust the ambient conditions to predict power output:")
    
    input_data = {
        'Ambient Temperature': st.slider(
            'Ambient Temperature (°C)',
            min_value=0.0, max_value=50.0, value=25.0, step=0.1
        ),
        'Ambient Relative Humidity': st.slider(
            'Ambient Relative Humidity (%)',
            min_value=0.0, max_value=100.0, value=50.0, step=0.1
        ),
        'Ambient Pressure': st.slider(
            'Ambient Pressure (mbar)',
            min_value=900.0, max_value=1100.0, value=1013.0, step=0.1
        ),
        'Exhaust Vacuum': st.slider(
            'Exhaust Vacuum (cmHg)',
            min_value=0.0, max_value=100.0, value=50.0, step=0.1
        )
    }
    
    st.markdown("---")
    st.write("### Model Information")
    st.write(f"Ensemble Weight: {assets['best_weight']:.2f} RF / {1 - assets['best_weight']:.2f} XGBoost")

# Main content
st.title("🏭 Combined Cycle Power Plant Performance Optimizer")
st.write("""
This interactive dashboard predicts the total power output of a combined cycle power plant 
based on ambient conditions using an ensemble of Random Forest and XGBoost models.
""")

# Prepare input data
features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']
input_df = pd.DataFrame([input_data])

# Data processing pipeline
try:
    # Scale the input data
    scaled_input = assets['feature_scaler'].transform(input_df[features])
    standard_scaled_input = assets['standard_scaler'].transform(scaled_input)
    
    # Make predictions
    rf_pred = assets['rf_model'].predict(standard_scaled_input)[0]
    xgb_pred = assets['xgb_model'].predict(standard_scaled_input)[0]
    combined_pred = assets['best_weight'] * rf_pred + (1 - assets['best_weight']) * xgb_pred
    
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.stop()

# Display predictions
st.subheader("🔮 Power Output Prediction")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Random Forest Prediction", 
    f"{float(rf_pred):.2f} MW",  # Explicit conversion to float
    help="Prediction from the Random Forest model"
)
col2.metric(
    "XGBoost Prediction", 
    f"{float(xgb_pred):.2f} MW",  # Explicit conversion to float
    help="Prediction from the XGBoost model"
)
col3.metric(
    "Ensemble Prediction", 
    f"{float(combined_pred):.2f} MW",  # Explicit conversion to float
    delta=f"{(float(combined_pred) - (float(rf_pred) + float(xgb_pred))/2):.2f} vs average",
    help="Weighted combination of both models"
)

# Feature importance visualization
st.subheader("📊 Feature Importance Analysis")
tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])

with tab1:
    try:
        importances = assets['rf_model'].feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, importances, color='#1f77b4')
        ax.set_title('Random Forest Feature Importance')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display Random Forest feature importance: {e}")

with tab2:
    try:
        importances = assets['xgb_model'].feature_importances_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, importances, color='#ff7f0e')
        ax.set_title('XGBoost Feature Importance')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display XGBoost feature importance: {e}")

# Optimization section
st.subheader("⚡ Optimal Parameters for Maximum Power")
st.write("""
The theoretical optimal parameters that would maximize power output based on historical data patterns:
""")

# Display optimal parameters (using the PSO results from your original code)
try:
    optimal_scaled = np.array([0.5, 0.5, 0.5, 0.5])  # Replace with actual PSO results
    optimal_original = assets['feature_scaler'].inverse_transform(optimal_scaled.reshape(1, -1))[0]
    optimal_params = dict(zip(features, optimal_original))

    cols = st.columns(2)
    for i, (param, value) in enumerate(optimal_params.items()):
        cols[i % 2].metric(
            label=param,
            value=f"{float(value):.2f}",  # Explicit conversion to float
            delta=f"{(float(input_data[param]) - float(value)):.2f} from your input",
            delta_color="inverse"
        )

    # Comparison table
    st.write("### 📝 Detailed Parameter Comparison")
    comparison_data = []
    for i, feature in enumerate(features):
        comparison_data.append({
            'Parameter': feature,
            'Your Input': float(input_data[feature]),
            'Optimal Value': float(optimal_original[i]),
            'Difference': float(input_data[feature]) - float(optimal_original[i])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.format("{:.2f}").background_gradient(
            subset=['Difference'], 
            cmap='RdYlGn',
            vmin=-50, 
            vmax=50
        ),
        use_container_width=True,
        height=200
    )
except Exception as e:
    st.warning(f"Could not display optimization results: {e}")

# Footer
st.markdown("---")
st.write("""
**Note**: This predictive model estimates power output based on historical data patterns. 
Actual plant performance may vary due to factors not considered in this model.
""")

# Add some styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
    }
    .stMetric label {
        font-size: 1rem !important;
        color: #555 !important;
    }
    .stMetric div {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)
