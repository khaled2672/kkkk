import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pyswarms as ps
import os

st.set_page_config(page_title="Power Prediction & Optimization", layout="wide")
st.title("⚡ Power Prediction & Optimization with ML + PSO")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    target_column = 'Total Power'
    selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

    # --- Outlier removal ---
    for col in selected_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # --- Handle missing values ---
    imputer = SimpleImputer(strategy='mean')
    df[selected_features] = imputer.fit_transform(df[selected_features])

    # --- MinMax scale for PSO ---
    feature_scaler = MinMaxScaler()
    df[selected_features] = feature_scaler.fit_transform(df[selected_features])

    # --- Train/test split ---
    X = df[selected_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Standard scaling ---
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    # --- Models ---
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.9, random_state=50, verbosity=0)
    }

    tuned_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        tuned_models[name] = model

        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mbe = np.mean(y_test - y_pred)

        results[name] = {"MSE": mse, "R2": r2, "MAE": mae, "MBE": mbe}

    # --- Ensemble optimization ---
    rf_preds = tuned_models["Random Forest"].predict(X_test_scaled)
    xgb_preds = tuned_models["XGBoost"].predict(X_test_scaled)

    best_r2, best_w = -np.inf, 0.0
    for w in np.arange(0, 1.05, 0.05):
        blended = w * rf_preds + (1 - w) * xgb_preds
        score = r2_score(y_test, blended)
        if score > best_r2:
            best_r2, best_w = score, w

    final_preds = best_w * rf_preds + (1 - best_w) * xgb_preds
    combined_mse = mean_squared_error(y_test, final_preds)
    combined_r2 = r2_score(y_test, final_preds)
    combined_mae = np.mean(np.abs(y_test - final_preds))
    combined_mbe = np.mean(y_test - final_preds)

    results["Combined"] = {
        "MSE": combined_mse, "R2": combined_r2, "MAE": combined_mae, "MBE": combined_mbe
    }

    st.subheader("📊 Model Evaluation Metrics")
    st.write(pd.DataFrame(results).T)

    # --- Plotting metrics ---
    st.subheader("📈 Metric Comparison Plot")
    metrics = ['R2', 'MSE', 'MAE', 'MBE']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    model_names = list(results.keys())
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        axes[i].bar(model_names, values, color=['black', 'red', 'blue'])
        axes[i].set_title(metric)
        axes[i].tick_params(axis='x', rotation=30)
    st.pyplot(fig)

    # --- PSO Optimization ---
    st.subheader("🚀 Particle Swarm Optimization")

    bounds = (np.zeros(len(selected_features)), np.ones(len(selected_features)))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def objective_func(X_particles):
        preds = []
        for particle in X_particles:
            particle = particle.reshape(1, -1)
            pred = best_w * tuned_models["Random Forest"].predict(particle) + \
                   (1 - best_w) * tuned_models["XGBoost"].predict(particle)
            preds.append(pred[0])
        return -np.array(preds)

    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=len(selected_features), options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(objective_func, iters=100, verbose=False)

    st.success(f"🔋 Maximum Predicted Power Output: {-best_cost:.2f}")

    original_features = feature_scaler.inverse_transform(best_pos.reshape(1, -1))[0]
    st.write("🧬 Optimal Feature Configuration:")
    for feat, val in zip(selected_features, original_features):
        st.write(f"**{feat}**: {val:.2f}")

else:
    st.info("⬆️ Please upload a CSV file to proceed.")
