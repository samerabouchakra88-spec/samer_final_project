import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json

# ==========================================
# 1. إعدادات واجهة التطبيق
# ==========================================
st.set_page_config(page_title="تنبؤ إنتاج زيت الزيتون", page_icon="🫒", layout="wide")

st.title("🫒 تطبيق الذكاء الاصطناعي لتنبؤ إنتاج زيت الزيتون")
st.markdown("### منطقة الشوف - مشروع التخرج")
st.write("أدخل بيانات الحقل والمناخ في القائمة الجانبية، وسيقوم الذكاء الاصطناعي بحساب الإنتاج المتوقع بدقة 98.6%.")

# ==========================================
# 2. تحميل النموذج وأسماء الأعمدة
# ==========================================
@st.cache_resource
def load_model_and_features():
    # Use xgb.Booster for compatibility across all xgboost versions
    booster = xgb.Booster()
    booster.load_model("olive_oil_xgb_model.json")

    try:
        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)
    except Exception:
        # Fallback to the known 30 features if file is missing
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
    st.success("✅ النموذج محمّل بنجاح — جاهز للتنبؤ")
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل النموذج: {e}")
    st.stop()

# ==========================================
# 3. القائمة الجانبية (مدخلات المزارع)
# ==========================================
st.sidebar.header("🌿 بيانات الحقل")

st.sidebar.subheader("1. بيانات الحقل الأساسية")
altitude = st.sidebar.number_input("الارتفاع عن سطح البحر (م)", 300, 1500, 800)
field_mean = st.sidebar.number_input("متوسط الإنتاج التاريخي للحقل (كغ/هكتار)", 500, 3000, 1500)
prev_oil = st.sidebar.number_input("إنتاج الزيت في السنة السابقة (كغ/هكتار)", 0, 3000, 1200)
prev_olives = st.sidebar.number_input("إنتاج الزيتون في السنة السابقة (كغ/هكتار)", 0, 15000, 8000)

st.sidebar.subheader("2. بيانات المناخ (هذا العام)")
rain = st.sidebar.number_input("الأمطار السنوية (مم)", 500, 2000, 1100)
spring_rain = st.sidebar.number_input("أمطار الربيع (مم)", 0, 500, 250)
temp = st.sidebar.number_input("متوسط حرارة الصيف (مئوية)", 20.0, 40.0, 32.0)
frost = st.sidebar.number_input("عدد أيام الصقيع", 0, 50, 15)

st.sidebar.subheader("3. الإدارة الزراعية")
trees = st.sidebar.number_input("عدد الأشجار في الهكتار", 100, 500, 250)
age = st.sidebar.number_input("عمر الأشجار (سنوات)", 5, 100, 30)
irrigation = st.sidebar.number_input("الري التكميلي (لتر/شجرة)", 0, 2000, 500)
fertilizer = st.sidebar.number_input("السماد (كغ/شجرة)", 0.0, 10.0, 3.0)
pest = st.sidebar.slider("مؤشر الآفات (0=سليم, 10=إصابة شديدة)", 0, 10, 2)
soil_type = st.sidebar.selectbox("نوع التربة", ["Loam", "Clay", "Sandy_Loam"])
pruning = st.sidebar.selectbox("شدة التقليم", ["Light", "Moderate", "Heavy"])

# ==========================================
# 4. حساب هندسة الميزات (Feature Engineering)
# ==========================================
if st.button("🚀 توقع الإنتاج الآن", use_container_width=True):
    with st.spinner("جاري تحليل البيانات المعقدة..."):
        pruning_dict = {"Light": 0, "Moderate": 1, "Heavy": 2}
        soil_dict = {"Clay": 0, "Loam": 1, "Sandy_Loam": 2}

        # Calculate engineered features
        heat_stress = temp * frost
        temp_rain_idx = temp / (rain / 100) if rain > 0 else temp
        spring_ratio = spring_rain / rain if rain > 0 else 0
        frost_alt = frost * (altitude / 1000)
        total_water = rain + irrigation
        water_per_tree = total_water / trees if trees > 0 else 0
        irr_rain_ratio = irrigation / (rain + 1)
        tree_prod = trees * age
        fert_per_tree = fertilizer * trees
        age_dens_ratio = age / trees if trees > 0 else 0
        yoy_change = (prev_oil - field_mean) / field_mean if field_mean > 0 else 0
        oil_vs_mean = prev_oil / field_mean if field_mean > 0 else 1.0

        # Create dictionary of all features
        feature_values = {
            'Altitude_m': altitude,
            'Annual_Rainfall_mm_year': rain,
            'Spring_Rainfall_mm_season': spring_rain,
            'Summer_Max_Temp_C_avg': temp,
            'Frost_Days_count': frost,
            'Irrigation_L_per_tree': irrigation,
            'Tree_Age_Years': age,
            'Tree_Density_per_hectare': trees,
            'Fertilizer_20_20_20_kg_per_tree': fertilizer,
            'Pest_Pressure_Index': pest,
            'Soil_Type_enc': soil_dict[soil_type],
            'Pruning_Intensity_enc': pruning_dict[pruning],
            'Prev_Year_Oil': prev_oil,
            'Prev_Year_Olives': prev_olives,
            'Oil_YoY_Change': yoy_change,
            'Rain_3yr_avg': rain,
            'Rain_2yr_avg': rain,
            'Irr_3yr_avg': irrigation,
            'Heat_Stress': heat_stress,
            'Temp_Rain_Index': temp_rain_idx,
            'Spring_Rain_Ratio': spring_ratio,
            'Frost_Altitude': frost_alt,
            'Total_Water': total_water,
            'Water_per_Tree': water_per_tree,
            'Irr_Rain_Ratio': irr_rain_ratio,
            'Tree_Productivity': tree_prod,
            'Fert_per_Tree': fert_per_tree,
            'Age_Density_Ratio': age_dens_ratio,
            'Field_Oil_Mean': field_mean,
            'Oil_vs_Field_Mean': oil_vs_mean
        }

        # Build dataframe with EXACT column order expected by model
        features_df = pd.DataFrame([feature_values], columns=feature_names)

        # ==========================================
        # 5. التنبؤ باستخدام xgb.Booster
        # ==========================================
        dmatrix = xgb.DMatrix(features_df, feature_names=feature_names)
        prediction = float(model.predict(dmatrix)[0])
        prediction = max(0, prediction)  # لا يمكن أن يكون الإنتاج سالباً

        st.markdown("---")
        st.subheader("🎯 النتيجة المتوقعة")
        col1, col2, col3 = st.columns(3)
        col1.metric("الإنتاج المتوقع لزيت الزيتون", f"{prediction:,.1f} كغ/هكتار")
        col2.metric("مقارنة بمتوسط الحقل", f"{(prediction / field_mean) * 100:.1f}%")
        col3.metric("مؤشر الإجهاد الحراري", f"{heat_stress:.1f}")

        # تفسير النتيجة
        st.markdown("---")
        st.subheader("📊 تفسير النتيجة")
        if prediction >= field_mean * 1.1:
            st.success(f"🌟 موسم ممتاز! الإنتاج المتوقع أعلى من المتوسط بنسبة {((prediction/field_mean)-1)*100:.1f}%")
        elif prediction >= field_mean * 0.9:
            st.info(f"✅ موسم جيد. الإنتاج المتوقع قريب من المتوسط التاريخي للحقل.")
        else:
            st.warning(f"⚠️ موسم أقل من المتوسط. الإنتاج المتوقع أقل من المتوسط بنسبة {(1-(prediction/field_mean))*100:.1f}%")

        # جدول ملخص المدخلات
        st.markdown("---")
        st.subheader("📋 ملخص بيانات الحقل المُدخلة")
        summary_data = {
            "العامل": ["الارتفاع", "الأمطار السنوية", "درجة حرارة الصيف", "أيام الصقيع",
                       "الري", "عمر الأشجار", "كثافة الأشجار", "التقليم", "نوع التربة"],
            "القيمة": [f"{altitude} م", f"{rain} مم", f"{temp}°C", f"{frost} يوم",
                       f"{irrigation} ل/شجرة", f"{age} سنة", f"{trees} شجرة/هكتار",
                       pruning, soil_type]
        }
        st.table(pd.DataFrame(summary_data))

        st.success("تم حساب النتيجة بناءً على 30 عاملاً زراعياً ومناخياً دقيقاً باستخدام نموذج XGBoost.")

# ==========================================
# 6. معلومات النموذج
# ==========================================
st.markdown("---")
with st.expander("ℹ️ معلومات عن النموذج"):
    st.markdown("""
    **نموذج XGBoost لتنبؤ إنتاج زيت الزيتون - منطقة الشوف، لبنان**
    
    | المعيار | القيمة |
    |---------|--------|
    | دقة النموذج (R²) | 98.6% |
    | متوسط الخطأ المطلق (MAE) | ±79.5 كغ/هكتار |
    | جذر متوسط مربع الخطأ (RMSE) | ±116.4 كغ/هكتار |
    | عدد الميزات | 30 ميزة |
    | حجم بيانات التدريب | 180 حقل-موسم |
    | الخوارزمية | XGBoost (Gradient Boosting) |
    
    **مشروع التخرج** — تطبيق الذكاء الاصطناعي في الزراعة اللبنانية
    """)
