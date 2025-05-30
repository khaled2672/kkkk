import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error
import pyswarms as ps

# --- Load Models and Scalers ---
rf_model = joblib.load("rf_model.joblib")
xgb_model = joblib.load("xgb_model.joblib")
standard_scaler = joblib.load("standard_scaler.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")
with open("best_weight.txt", "r") as f:
    best_w = float(f.read().strip())

selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

st.set_page_config(layout="wide")
st.title("🔧 Power Prediction & Optimization App")

# --- Section: Upload & View Dataset ---
st.header("📥 Upload and Preview Data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target_column = 'Total Power'

    # Preprocessing steps
    # --- Outlier removal ---
    for col in selected_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # --- Missing value imputation ---
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[selected_features] = imputer.fit_transform(df[selected_features])

    # --- Feature scaling for PSO ---
    df_scaled = df.copy()
    df_scaled[selected_features] = feature_scaler.transform(df[selected_features])
    X_scaled = standard_scaler.transform(df[selected_features])
    y = df[target_column]

    # --- Predictions ---
    rf_preds = rf_model.predict(X_scaled)
    xgb_preds = xgb_model.predict(X_scaled)
    final_preds = best_w * rf_preds + (1 - best_w) * xgb_preds

    # --- Metrics ---
    mse = mean_squared_error(y, final_preds)
    r2 = r2_score(y, final_preds)
    mae = np.mean(np.abs(y - final_preds))
    mbe = np.mean(y - final_preds)

    st.subheader("📊 Model Performance")
    st.write(f"**R2 Score**: {r2:.4f}")
    st.write(f"**MSE**: {mse:.4f}")
    st.write(f"**MAE**: {mae:.4f}")
    st.write(f"**MBE**: {mbe:.4f}")

    # --- Plot actual vs predicted ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.values, label='Actual', alpha=0.7)
    ax.plot(final_preds, label='Predicted', alpha=0.7)
    ax.set_title("Actual vs Predicted Power")
    ax.legend()
    st.pyplot(fig)

# --- Section: PSO Optimization ---
st.header("🚀 Optimize for Maximum Power Output")

if st.button("Run PSO Optimization"):
    bounds = (np.zeros(len(selected_features)), np.ones(len(selected_features)))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def objective_func(X):
        preds = []
        for particle in X:
            particle = particle.reshape(1, -1)
            pred = best_w * rf_model.predict(particle) + (1 - best_w) * xgb_model.predict(particle)
            preds.append(pred[0])
        return -np.array(preds)

    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=len(selected_features), options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(objective_func, iters=100, verbose=False)

    st.success(f"✅ Max predicted power: {-best_cost:.2f}")

    st.subheader("🔍 Optimal Scaled Features")
    st.write(dict(zip(selected_features, best_pos.round(4))))

    original_features = feature_scaler.inverse_transform(best_pos.reshape(1, -1))[0]
    st.subheader("📈 Optimal Original Feature Values")
    for feat, val in zip(selected_features, original_features):
        st.write(f"**{feat}**: {val:.4f}")
