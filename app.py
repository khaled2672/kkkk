import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os

st.set_page_config(page_title="Power Prediction App", layout="wide")
st.title("🔌 Power Prediction using Ensemble Models")

# File upload
uploaded_file = st.file_uploader("Upload CSV data file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded and read successfully!")

    # Data preparation
    target_column = 'Total Power'
    selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

    # Remove outliers
    for col in selected_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Interaction feature
    df['Temp*Humidity'] = df['Ambient Temperature'] * df['Ambient Relative Humidity']
    selected_features.append('Temp*Humidity')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[selected_features] = imputer.fit_transform(df[selected_features])

    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df[selected_features])

    # Scaling
    feature_scaler = MinMaxScaler()
    X_poly = feature_scaler.fit_transform(X_poly)

    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)
    joblib.dump(standard_scaler, "standard_scaler.pkl")

    # Models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.9, random_state=50, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.9, random_state=50),
        "CatBoost": CatBoostRegressor(iterations=300, depth=10, learning_rate=0.1, verbose=0, random_state=50),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10)
    }

    st.subheader("🔧 Training Models...")
    results = {}
    tuned_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        tuned_models[name] = model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mbe = np.mean(y_test - y_pred)

        results[name] = {"MSE": mse, "R2": r2, "MAE": mae, "MBE": mbe}
        st.write(f"✅ {name} — R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, MBE: {mbe:.4f}")

    # Weighted Ensemble
    rf_preds = tuned_models["Random Forest"].predict(X_test_scaled)
    xgb_preds = tuned_models["XGBoost"].predict(X_test_scaled)

    best_r2, best_w = -np.inf, 0.0
    for w in np.arange(0, 1.05, 0.05):
        blended = w * rf_preds + (1 - w) * xgb_preds
        score = r2_score(y_test, blended)
        if score > best_r2:
            best_r2, best_w = score, w

    final_preds = best_w * rf_preds + (1 - best_w) * xgb_preds
    results["Combined"] = {
        "MSE": mean_squared_error(y_test, final_preds),
        "R2": r2_score(y_test, final_preds),
        "MAE": np.mean(np.abs(y_test - final_preds)),
        "MBE": np.mean(y_test - final_preds)
    }

    # Save models
    joblib.dump(tuned_models["Random Forest"], "rf_model.pkl", compress=("zlib", 9))
    joblib.dump(tuned_models["XGBoost"], "xgb_model.pkl", compress=("zlib", 3))
    with open("best_weight.txt", "w") as f:
        f.write(str(best_w))

    st.subheader("📈 Ensemble Model Results")
    st.write(f"✅ Best Ensemble Weight: {best_w:.2f} RF / {1 - best_w:.2f} XGB")
    st.write(f"Combined R2: {results['Combined']['R2']:.4f}")
    st.write(f"MSE: {results['Combined']['MSE']:.4f}")
    st.write(f"MAE: {results['Combined']['MAE']:.4f}")
    st.write(f"MBE: {results['Combined']['MBE']:.4f}")

    # Input form for prediction
    st.subheader("🔍 Predict Total Power")
    with st.form("predict_form"):
        temp = st.number_input("Ambient Temperature")
        humidity = st.number_input("Ambient Relative Humidity")
        pressure = st.number_input("Ambient Pressure")
        vacuum = st.number_input("Exhaust Vacuum")
        submit = st.form_submit_button("Predict")

    if submit:
        # Create features
        input_df = pd.DataFrame([{
            'Ambient Temperature': temp,
            'Ambient Relative Humidity': humidity,
            'Ambient Pressure': pressure,
            'Exhaust Vacuum': vacuum,
            'Temp*Humidity': temp * humidity
        }])

        input_poly = poly.transform(input_df)
        input_scaled = feature_scaler.transform(input_poly)
        input_final = standard_scaler.transform(input_scaled)

        # Load models
        rf_model = joblib.load("rf_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")
        with open("best_weight.txt", "r") as f:
            w = float(f.read().strip())

        prediction = w * rf_model.predict(input_final) + (1 - w) * xgb_model.predict(input_final)
        st.success(f"🔋 Predicted Total Power: {prediction[0]:.2f}")

else:
    st.info("📂 Please upload a CSV file to begin.")

