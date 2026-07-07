import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Olive Oil Yield Predictor",
    page_icon="🫒",
    layout="wide"
)

# Header
st.title("🫒 Olive Oil Yield Predictor")
st.markdown("### Shouf Region, Lebanon — AI-Powered Harvest Forecasting")
st.markdown(
    "Fill in your field details on the left, and our AI model will instantly "
    "estimate how many kilograms of olive oil your grove will produce this season."
)
st.markdown("---")

# ==========================================
# 2. Load Model
# ==========================================
@st.cache_resource
def load_model_and_features():
    booster = xgb.Booster()
    booster.load_model("olive_oil_xgb_model.json")
    try:
        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)
    except Exception:
        feature_names = [
            'Altitude_m', 'Annual_Rainfall_mm_year', 'Spring_Rainfall_mm_season',
            'Summer_Max_Temp_C_avg', 'Frost_Days_count', 'Irrigation_L_per_tree',
            'Tree_Age_Years', 'Tree_Density_per_hectare', 'Fertilizer_20_20_20_kg_per_tree',
            'Pest_Pressure_Index', 'Soil_Type_enc', 'Pruning_Intensity_enc',
            'Prev_Year_Oil', 'Prev_Year_Olives', 'Oil_YoY_Change', 'Rain_3yr_avg',
            'Rain_2yr_avg', 'Irr_3yr_avg', 'Heat_Stress', 'Temp_Rain_Index',
            'Spring_Rain_Ratio', 'Frost_Altitude', 'Total_Water', 'Water_per_Tree',
            'Irr_Rain_Ratio', 'Tree_Productivity', 'Fert_per_Tree', 'Age_Density_Ratio',
            'Field_Oil_Mean', 'Oil_vs_Field_Mean'
        ]
    return booster, feature_names

try:
    model, feature_names = load_model_and_features()
    st.success("✅ AI model loaded and ready!")
except Exception as e:
    st.error(f"⚠️ Could not load the model: {e}")
    st.stop()

# ==========================================
# 3. Sidebar — Farmer Inputs
# ==========================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Olive_oil_from_Oneglia.jpg/320px-Olive_oil_from_Oneglia.jpg",
    use_container_width=True
)
st.sidebar.title("🌿 Your Field Details")
st.sidebar.markdown("*Fill in the details below about your grove and this season's conditions.*")

# --- Section 1: Field Info ---
st.sidebar.markdown("### 🏡 Field Information")
altitude = st.sidebar.number_input(
    "Altitude above sea level (m)",
    min_value=300, max_value=1500, value=800,
    help="How high is your field above sea level?"
)
field_mean = st.sidebar.number_input(
    "Your field's average historical oil yield (kg/ha)",
    min_value=500, max_value=3000, value=1500,
    help="What is the typical olive oil yield for this field in a normal year?"
)
prev_oil = st.sidebar.number_input(
    "Last year's oil yield (kg/ha)",
    min_value=0, max_value=3000, value=1200,
    help="How much olive oil did your field produce last season?"
)
prev_olives = st.sidebar.number_input(
    "Last year's olive harvest (kg/ha)",
    min_value=0, max_value=15000, value=8000,
    help="Total weight of olives harvested last year."
)

# --- Section 2: Weather ---
st.sidebar.markdown("### 🌦️ This Season's Weather")
rain = st.sidebar.number_input(
    "Annual rainfall (mm)",
    min_value=500, max_value=2000, value=1100,
    help="Total rainfall received this year."
)
spring_rain = st.sidebar.number_input(
    "Spring rainfall (mm)",
    min_value=0, max_value=500, value=250,
    help="Rainfall during spring months only."
)
temp = st.sidebar.number_input(
    "Average summer temperature (°C)",
    min_value=20.0, max_value=40.0, value=32.0,
    help="Average daytime temperature during summer."
)
frost = st.sidebar.number_input(
    "Number of frost days",
    min_value=0, max_value=50, value=15,
    help="How many days did temperatures drop below 0°C?"
)

# --- Section 3: Farm Management ---
st.sidebar.markdown("### 🚜 Farm Management")
trees = st.sidebar.number_input(
    "Trees per hectare",
    min_value=100, max_value=500, value=250,
    help="How many olive trees are planted per hectare?"
)
age = st.sidebar.number_input(
    "Average tree age (years)",
    min_value=5, max_value=100, value=30,
    help="How old are your olive trees on average?"
)
irrigation = st.sidebar.number_input(
    "Supplemental irrigation (liters/tree)",
    min_value=0, max_value=2000, value=500,
    help="How much extra water did you provide per tree?"
)
fertilizer = st.sidebar.number_input(
    "Fertilizer applied (kg/tree)",
    min_value=0.0, max_value=10.0, value=3.0,
    help="Amount of 20-20-20 fertilizer used per tree."
)
pest = st.sidebar.slider(
    "Pest pressure level (0 = healthy, 10 = severe infestation)",
    min_value=0, max_value=10, value=2,
    help="Rate the pest damage in your grove this season."
)
soil_type = st.sidebar.selectbox(
    "Soil type",
    ["Loam", "Clay", "Sandy_Loam"],
    help="What type of soil does your grove have?"
)
pruning = st.sidebar.selectbox(
    "Pruning intensity",
    ["Light", "Moderate", "Heavy"],
    help="How heavily were the trees pruned this year?"
)

# ==========================================
# 4. Predict Button
# ==========================================
st.markdown("### 👇 Ready? Click below to get your harvest forecast!")
predict_btn = st.button("🚀 Predict My Olive Oil Yield", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("🔍 Analyzing 30 field and climate factors..."):

        # Encode categorical inputs
        pruning_dict = {"Light": 0, "Moderate": 1, "Heavy": 2}
        soil_dict    = {"Clay": 0, "Loam": 1, "Sandy_Loam": 2}

        # Compute engineered features
        heat_stress    = temp * frost
        temp_rain_idx  = temp / (rain / 100) if rain > 0 else temp
        spring_ratio   = spring_rain / rain if rain > 0 else 0
        frost_alt      = frost * (altitude / 1000)
        total_water    = rain + irrigation
        water_per_tree = total_water / trees if trees > 0 else 0
        irr_rain_ratio = irrigation / (rain + 1)
        tree_prod      = trees * age
        fert_per_tree  = fertilizer * trees
        age_dens_ratio = age / trees if trees > 0 else 0
        yoy_change     = (prev_oil - field_mean) / field_mean if field_mean > 0 else 0
        oil_vs_mean    = prev_oil / field_mean if field_mean > 0 else 1.0

        feature_values = {
            'Altitude_m':                      altitude,
            'Annual_Rainfall_mm_year':         rain,
            'Spring_Rainfall_mm_season':       spring_rain,
            'Summer_Max_Temp_C_avg':           temp,
            'Frost_Days_count':                frost,
            'Irrigation_L_per_tree':           irrigation,
            'Tree_Age_Years':                  age,
            'Tree_Density_per_hectare':        trees,
            'Fertilizer_20_20_20_kg_per_tree': fertilizer,
            'Pest_Pressure_Index':             pest,
            'Soil_Type_enc':                   soil_dict[soil_type],
            'Pruning_Intensity_enc':           pruning_dict[pruning],
            'Prev_Year_Oil':                   prev_oil,
            'Prev_Year_Olives':                prev_olives,
            'Oil_YoY_Change':                  yoy_change,
            'Rain_3yr_avg':                    rain,
            'Rain_2yr_avg':                    rain,
            'Irr_3yr_avg':                     irrigation,
            'Heat_Stress':                     heat_stress,
            'Temp_Rain_Index':                 temp_rain_idx,
            'Spring_Rain_Ratio':               spring_ratio,
            'Frost_Altitude':                  frost_alt,
            'Total_Water':                     total_water,
            'Water_per_Tree':                  water_per_tree,
            'Irr_Rain_Ratio':                  irr_rain_ratio,
            'Tree_Productivity':               tree_prod,
            'Fert_per_Tree':                   fert_per_tree,
            'Age_Density_Ratio':               age_dens_ratio,
            'Field_Oil_Mean':                  field_mean,
            'Oil_vs_Field_Mean':               oil_vs_mean,
        }

        features_df = pd.DataFrame([feature_values], columns=feature_names)
        dmatrix     = xgb.DMatrix(features_df, feature_names=feature_names)
        prediction  = float(model.predict(dmatrix)[0])
        prediction  = max(0, prediction)

        # ==========================================
        # 5. Results
        # ==========================================
        st.markdown("---")
        st.markdown("## 🎯 Your Harvest Forecast")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="🫒 Predicted Oil Yield",
            value=f"{prediction:,.0f} kg/ha",
            delta=f"{prediction - field_mean:+,.0f} vs. your average"
        )
        col2.metric(
            label="📊 Compared to Your Field Average",
            value=f"{(prediction / field_mean) * 100:.1f}%"
        )
        col3.metric(
            label="🌡️ Heat Stress Index",
            value=f"{heat_stress:.0f}",
            help="Higher = more heat stress on trees (temp × frost days)"
        )

        # Season verdict
        st.markdown("---")
        st.markdown("### 🌾 Season Outlook")
        pct_diff = ((prediction / field_mean) - 1) * 100
        if prediction >= field_mean * 1.1:
            st.success(
                f"🌟 **Excellent season ahead!** Your predicted yield is "
                f"**{pct_diff:+.1f}% above** your field's historical average. "
                f"Conditions look great — keep up the good work!"
            )
        elif prediction >= field_mean * 0.9:
            st.info(
                f"✅ **Good season expected.** Your predicted yield is close to "
                f"your field's normal average ({pct_diff:+.1f}%). "
                f"No major concerns — standard care should be sufficient."
            )
        else:
            st.warning(
                f"⚠️ **Below-average season expected.** Your predicted yield is "
                f"**{abs(pct_diff):.1f}% below** your field's average. "
                f"Consider reviewing irrigation, pest control, or fertilization."
            )

        # Farmer tips based on inputs
        st.markdown("---")
        st.markdown("### 💡 Personalized Tips for Your Grove")
        tips = []
        if pest >= 6:
            tips.append("🐛 **High pest pressure detected** — consider a targeted treatment before harvest.")
        if irrigation < 200:
            tips.append("💧 **Low irrigation** — supplemental watering during dry spells can boost yield significantly.")
        if fertilizer < 2:
            tips.append("🌱 **Low fertilizer use** — increasing nutrients may improve oil content and yield.")
        if frost > 30:
            tips.append("❄️ **High frost exposure** — monitor young trees for frost damage.")
        if pruning == "Heavy":
            tips.append("✂️ **Heavy pruning** can reduce yield this season but improves long-term tree health.")
        if not tips:
            tips.append("👍 Your management practices look solid! Keep monitoring weather conditions.")
        for tip in tips:
            st.markdown(f"- {tip}")

        # Input summary table
        st.markdown("---")
        st.markdown("### 📋 Summary of Your Inputs")
        summary = pd.DataFrame({
            "Factor":       ["Altitude", "Annual Rainfall", "Summer Temp", "Frost Days",
                             "Irrigation", "Tree Age", "Tree Density", "Pruning", "Soil Type",
                             "Pest Level", "Fertilizer"],
            "Your Value":   [f"{altitude} m", f"{rain} mm", f"{temp}°C", f"{frost} days",
                             f"{irrigation} L/tree", f"{age} years", f"{trees} trees/ha",
                             pruning, soil_type, f"{pest}/10", f"{fertilizer} kg/tree"]
        })
        st.table(summary)

# ==========================================
# 6. Footer — Model Info
# ==========================================
st.markdown("---")
with st.expander("ℹ️ About the AI Model"):
    st.markdown("""
    **XGBoost Model — Olive Oil Yield Prediction | Shouf Region, Lebanon**

    | Metric | Value |
    |--------|-------|
    | Model Accuracy (R²) | 98.6% |
    | Mean Absolute Error (MAE) | ±79.5 kg/ha |
    | Root Mean Squared Error (RMSE) | ±116.4 kg/ha |
    | Number of Features | 30 agricultural & climate factors |
    | Training Data | 200 field-seasons |
    | Algorithm | XGBoost (Gradient Boosting) |

    This model was trained on real field data from the Shouf region and uses 30 factors
    including climate, soil, tree characteristics, and farm management practices.

    **Final Year Project** — Applying AI in Lebanese Agriculture 🌿
    """)

st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Built with ❤️ for Lebanese olive farmers | Shouf Region AI Agriculture Project"
    "</div>",
    unsafe_allow_html=True
)
