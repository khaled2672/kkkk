import streamlit as st 
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown(
            """<style>
            .stApp {
                background-image: url("https://img1.wsimg.com/isteam/getty/2007232407/:/rs=w:1750,h:1000,cg:true,m/cr=w:1750,h:1000");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                color: white;
            }
            .stApp:before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0, 0, 0, 0.85);
                z-index: -1;
            }
            .main .block-container {
                background-color: rgba(0, 0, 0, 0.8);
                padding: 2rem;
                border-radius: 12px;
                backdrop-filter: blur(5px);
            }
            [data-testid="stSidebar"] > div:first-child {
                background-color: rgba(20, 20, 20, 0.95) !important;
                color: white;
            }
            .stDownloadButton>button, .stButton>button {
                background-color: black !important;
                color: white !important;
                border: 1px solid white !important;
                font-weight: bold;
            }
            .stDownloadButton>button:hover, .stButton>button:hover {
                background-color: #222 !important;
            }
            /* Sidebar slider labels white */
            .css-1aumxhk .stSlider > label {
                color: white !important;
                font-weight: bold;
            }
            /* File uploader label bold */
            .css-1aumxhk label[for="fileUploader"] {
                font-weight: bold;
                color: white !important;
            }
            </style>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<style>
            .stApp {
                background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                color: black;
            }
            .stApp:before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(255, 255, 255, 0.85);
                z-index: -1;
            }
            .main .block-container {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 12px;
                backdrop-filter: blur(4px);
            }
            [data-testid="stSidebar"] > div:first-child {
                background-color: rgba(250, 250, 250, 0.95) !important;
            }
            .stDownloadButton>button, .stButton>button {
                background-color: black !important;
                color: white !important;
                border: 1px solid black !important;
                font-weight: bold;
            }
            .stDownloadButton>button:hover, .stButton>button:hover {
                background-color: #222 !important;
            }
            /* Sidebar slider labels black */
            .css-1aumxhk .stSlider > label {
                color: black !important;
                font-weight: bold;
            }
            /* File uploader label bold black */
            .css-1aumxhk label[for="fileUploader"] {
                font-weight: bold;
                color: black !important;
            }
            </style>""",
            unsafe_allow_html=True
        )

@st.cache_resource
def load_models():
    try:
        return (
            joblib.load('rf_model.joblib'),
            joblib.load('xgb_model.joblib'),
            joblib.load('scaler.joblib')
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

def map_columns(df):
    column_mapping = {
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT", "Temperature (°C)"],
        "Ambient Relative Humidity": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
    }
    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[target] = name
                break
    return mapped_columns

@st.cache_data
def generate_example_csv():
    example_data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)
    st.subheader("How to Use")
    st.markdown("""  
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models  
    4. Upload CSV for batch predictions  
    """)
    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    feature_bounds = {
        'Ambient Temperature': [0.0, 50.0],
        'Ambient Relative Humidity': [10.0, 100.0],
        'Ambient Pressure': [799.0, 1035.0],
        'Exhaust Vacuum': [3.0, 12.0]
    }

    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in feature_bounds.items():
        default = (low + high) / 2
        inputs[feature] = st.slider(
            feature, low, high, default,
            help=f"Adjust {feature} between {low} and {high}"
        )

    if st.button("🔄 Reset to Defaults"):
        for feature in inputs:
            inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2

# Main
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

feature_names = list(feature_bounds.keys())
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = 0.65

with st.spinner("Making predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""<div style="background-color: rgba(0,0,0,0.7); padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="margin-top: 0; color: white;">Random Forest</h3>
            <h2 style="color: red;">{rf_pred:.2f} MW</h2>
        </div>""", unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""<div style="background-color: rgba(0,0,0,0.7); padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="margin-top: 0; color: white;">XGBoost</h3>
            <h2 style="color: red;">{xgb_pred:.2f} MW</h2>
        </div>""", unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""<div style="background-color: rgba(0,0,0,0.7); padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="margin-top: 0; color: white;">Ensemble (65% RF / 35% XGB)</h3>
            <h2 style="color: white;">{ensemble_pred:.2f} MW</h2>
            <p style="margin-bottom: 0; font-size: 0.9rem; color: white;">{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg</p>
        </div>""", unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("📁 Batch Prediction via CSV Upload")

# Bold label for file uploader
st.markdown("**Upload CSV file with plant conditions**")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        mapped_cols = map_columns(df)

        if len(mapped_cols) < len(feature_bounds):
            st.warning("Some required columns are missing or could not be mapped automatically. Please check the CSV headers.")
        else:
            pred_features = df[[mapped_cols[feat] for feat in feature_names]]
            pred_features.columns = feature_names

            scaled_batch_features = scaler.transform(pred_features)

            rf_preds = rf_model.predict(scaled_batch_features)
            xgb_preds = xgb_model.predict(scaled_batch_features)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds

            df["Random Forest Prediction (MW)"] = rf_preds
            df["XGBoost Prediction (MW)"] = xgb_preds
            df["Ensemble Prediction (MW)"] = final_preds

            # White success message for predictions
            st.markdown(f'<p style="color:white; font-weight:bold;">Predicted {len(final_preds)} records successfully!</p>', unsafe_allow_html=True)
            st.dataframe(df)

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="ccpp_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
else:
    st.markdown(
        """
        <div style="color: white; font-weight: bold; font-size: 16px;">
            Upload a CSV file to run batch predictions. You can download an example file below.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.download_button(
        label="Download Example CSV",
        data=generate_example_csv(),
        file_name="example_ccpp_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest (65%), XGBoost (35%)
""")
