import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Gas Turbine Power Prediction", layout="wide")
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output from ambient sensor inputs and optimize operating conditions.")

# === Load models and transformers ===
try:
    rf_model = joblib.load("rf_model.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
    minmax_scaler = joblib.load("minmax_scaler.joblib")
    standard_scaler = joblib.load("standard_scaler.joblib")
    poly = joblib.load("poly_transformer.joblib")
    best_weight = joblib.load("best_weight.txt")
except Exception as e:
    st.error(f"❌ Model/component loading failed: {str(e)}")
    st.stop()

# === Check PSO availability ===
try:
    from pyswarms.single.global_best import GlobalBestPSO
    pso_installed = True
except ModuleNotFoundError:
    pso_installed = False
    st.warning("⚠️ `pyswarms` is not installed. Optimization will be unavailable.")

# === Helper functions ===
def preprocess_input(features):
    poly_features = poly.transform(features)
    scaled = minmax_scaler.transform(poly_features)
    return standard_scaler.transform(scaled)

def predict_models(input_array):
    rf = rf_model.predict(input_array)[0]
    xgb = xgb_model.predict(input_array)[0]
    ensemble = best_weight * rf + (1 - best_weight) * xgb
    return rf, xgb, ensemble

# === Tabs ===
tabs = st.tabs(["🔍 Single Prediction", "📂 Predict from CSV", "🧠 Optimize with PSO"])

# === Tab 1: Manual Input ===
with tabs[0]:
    st.subheader("Manual Sensor Input")
    col1, col2 = st.columns(2)
    with col1:
        T = st.number_input("🌡️ Ambient Temperature (°C)", 0.0, 50.0, 25.0)
        AP = st.number_input("🌬️ Ambient Pressure (mbar)", 500.0, 1100.0, 1010.0)
    with col2:
        RH = st.number_input("💧 Relative Humidity (%)", 0.0, 100.0, 60.0)
        EV = st.number_input("🌪️ Exhaust Vacuum (cm Hg)", 0.0, 10.0, 4.5)

    if st.button("⚡ Predict Power Output"):
        features = np.array([[T, RH, AP, EV]])
        final_input = preprocess_input(features)
        rf_pred, xgb_pred, ensemble_pred = predict_models(final_input)

        st.markdown("### 📊 Prediction Results")
        pred_df = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "Ensemble"],
            "Power Output (MW)": [rf_pred, xgb_pred, ensemble_pred]
        })
        st.table(pred_df)

# === Tab 2: CSV Upload ===
with tabs[1]:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file containing ambient conditions. Required columns (with flexible naming):")

    column_mappings = {
        "T": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT", "T"],
        "RH": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "AP": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "V": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV", "V"],
    }

    def map_columns(df):
        rename_map = {}
        for std_col, aliases in column_mappings.items():
            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = std_col
                    break
        return df.rename(columns=rename_map)

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = map_columns(df)
            required_cols = ["T", "RH", "AP", "V"]

            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"❌ Missing required columns after mapping: {', '.join(missing)}")
            else:
                if df[required_cols].isnull().sum().sum() > 0:
                    st.warning("⚠️ Missing values detected. Dropping affected rows.")
                    df.dropna(subset=required_cols, inplace=True)

                inputs = preprocess_input(df[required_cols].values)
                rf_preds = rf_model.predict(inputs)
                xgb_preds = xgb_model.predict(inputs)
                ensemble_preds = best_weight * rf_preds + (1 - best_weight) * xgb_preds

                df["Predicted Power (MW)"] = ensemble_preds
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# === Tab 3: PSO Optimization ===
with tabs[2]:
    if not pso_installed:
        st.warning("⚠️ Install `pyswarms` to enable optimization.")
    else:
        st.subheader("🧠 Particle Swarm Optimization")
        st.markdown("Find optimal ambient conditions to **maximize** power output.")

        pso_bounds = {
            "Ambient Temperature": [15.0, 35.0],
            "Relative Humidity": [20.0, 80.0],
            "Ambient Pressure": [798.0, 802.0],
            "Exhaust Vacuum": [3.5, 7.0],
        }

        lb = np.array([v[0] for v in pso_bounds.values()])
        ub = np.array([v[1] for v in pso_bounds.values()])
        pso_features = list(pso_bounds.keys())

        def objective(x):
            preds = []
            for row in x:
                inp = preprocess_input(row.reshape(1, -1))
                _, _, ensemble = predict_models(inp)
                preds.append(-ensemble)  # Negative because PSO minimizes
            return np.array(preds)

        if st.button("🚀 Optimize Conditions"):
            with st.spinner("Running optimization..."):
                optimizer = GlobalBestPSO(
                    n_particles=30, dimensions=len(lb),
                    options={"c1": 0.5, "c2": 0.3, "w": 0.9},
                    bounds=(lb, ub)
                )
                cost, pos = optimizer.optimize(objective, iters=100)

                inp = preprocess_input(pos.reshape(1, -1))
                rf, xgb, ensemble = predict_models(inp)

                st.success(f"🎯 Optimized Power Output: {ensemble:.2f} MW")
                st.markdown("### 🌡️ Optimal Ambient Settings")
                opt_df = pd.DataFrame([pos], columns=pso_features).T
                opt_df.columns = ["Optimal Value"]
                st.table(opt_df)
