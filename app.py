import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load saved models & scaler
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

st.title("Power Output Prediction & Optimization")

st.markdown("""
This app predicts the power output based on environmental parameters using a Random Forest and XGBoost ensemble.  
You can adjust input features and ensemble weight to see the predicted power output.  
Additionally, optimize parameters using Particle Swarm Optimization (PSO) for max power.
""")

# Sidebar inputs
st.sidebar.header("Input Features")
input_data = {}
for feature in selected_features:
    if feature == 'Ambient Temperature':
        val = st.sidebar.slider(feature, 20.0, 30.0, 25.0)
    elif feature == 'Ambient Relative Humidity':
        val = st.sidebar.slider(feature, 40.0, 70.0, 55.0)
    elif feature == 'Ambient Pressure':
        val = st.sidebar.slider(feature, 799.0, 800.0, 799.5)
    elif feature == 'Exhaust Vacuum':
        val = st.sidebar.slider(feature, 4.5, 6.0, 5.0)
    input_data[feature] = val

ensemble_weight = st.sidebar.slider("Ensemble Weight for RF (0=only XGB, 1=only RF)", 0.0, 1.0, 0.5)

# Prepare input
input_df = pd.DataFrame([input_data])
# Scale input features
scaled_input = scaler.transform(input_df)

# Predict
rf_pred = rf_model.predict(scaled_input)[0]
xgb_pred = xgb_model.predict(scaled_input)[0]
ensemble_pred = ensemble_weight * rf_pred + (1 - ensemble_weight) * xgb_pred

st.subheader("Predicted Power Output")
st.write(f"Random Forest Prediction: {rf_pred:.2f} MW")
st.write(f"XGBoost Prediction: {xgb_pred:.2f} MW")
st.write(f"Ensemble Prediction: {ensemble_pred:.2f} MW")

# PSO Optimization button and logic
st.markdown("---")
st.subheader("Optimize Parameters for Maximum Power")

if st.button("Run PSO Optimization"):
    from pyswarms.single.global_best import GlobalBestPSO

    # Define bounds consistent with training
    feature_bounds = [
        (20.0, 30.0),     # Ambient Temperature
        (40.0, 70.0),     # Ambient Relative Humidity
        (799.0, 800.0),   # Ambient Pressure
        (4.5, 6.0),       # Exhaust Vacuum
        (0.0, 1.0)        # Ensemble weight
    ]

    def objective_function(x):
        features = x[:, :-1]
        weights = x[:, -1].reshape(-1, 1)
        scaled_features = scaler.transform(features)
        rf_p = rf_model.predict(scaled_features).reshape(-1, 1)
        xgb_p = xgb_model.predict(scaled_features).reshape(-1, 1)
        ensemble_p = weights * rf_p + (1 - weights) * xgb_p
        return -ensemble_p.flatten()  # Maximize power

    lb = np.array([b[0] for b in feature_bounds])
    ub = np.array([b[1] for b in feature_bounds])
    optimizer = GlobalBestPSO(n_particles=50, dimensions=len(lb), options={'c1':0.5,'c2':0.3,'w':0.9}, bounds=(lb, ub))
    cost, pos = optimizer.optimize(objective_function, iters=100)

    opt_features = pos[:-1]
    opt_weight = pos[-1]

    opt_df = pd.DataFrame([opt_features], columns=selected_features)
    opt_scaled = scaler.transform(opt_df)
    rf_opt_power = rf_model.predict(opt_scaled)[0]
    xgb_opt_power = xgb_model.predict(opt_scaled)[0]
    opt_power = opt_weight * rf_opt_power + (1 - opt_weight) * xgb_opt_power

    st.write("### Optimal Parameters:")
    for f, v in zip(selected_features, opt_features):
        st.write(f"{f}: {v:.2f}")

    st.write(f"Optimal Ensemble Weight for RF: {opt_weight:.2f}")
    st.write(f"Optimal Ensemble Weight for XGB: {1 - opt_weight:.2f}")
    st.write(f"Predicted Maximum Power: {opt_power:.2f} MW")

