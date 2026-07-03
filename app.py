import streamlit as st
import joblib
import numpy as np
import pandas as pd # Import pandas to create a DataFrame for prediction

# 1. Eğitilmiş Modeli Yükle
@st.cache_resource
def load_model():
    # Kendi kaydettiğin model dosyasının adını buraya yazmalısın
    return joblib.load("Model.pkl")

model = load_model()

# --- DİL AYARLARI ---
# Dil seçeneğini sidebar'da göster
language = st.sidebar.selectbox(
    "🌐 Language / Dil",
    ["Türkçe", "English"],
    index=0
)

# Dil paketleri
translations = {
    "Türkçe": {
        "title": "🏋️‍♂️ Yapay Zeka ile Vücut Yağ Oranı Hesaplayıcı",
        "description": "Lütfen aşağıdaki anatomik ölçülerinizi giriniz.",
        "age": "Yaş",
        "age_help": "Yaşınız (yıl)",
        "weight": "Kilo (kg)",
        "weight_help": "Kilonuz (kg)",
        "height": "Boy (cm)",
        "height_help": "Boyunuz (cm)",
        "neck": "Boyun Çevresi (cm)",
        "neck_help": "Boyun çevresi (cm)",
        "chest": "Göğüs Çevresi (cm)",
        "chest_help": "Göğüs çevresi (cm)",
        "abdomen": "Karın Çevresi (cm)",
        "abdomen_help": "Karın çevresi (cm) (göbek hizasından)",
        "hip": "Kalça Çevresi (cm)",
        "hip_help": "Kalça çevresi (cm)",
        "thigh": "Uyluk Çevresi (cm)",
        "thigh_help": "Uyluk çevresi (cm)",
        "knee": "Diz Çevresi (cm)",
        "knee_help": "Diz çevresi (cm)",
        "ankle": "Ayak Bileği Çevresi (cm)",
        "ankle_help": "Ayak bileği çevresi (cm)",
        "biceps": "Pazu Çevresi (cm)",
        "biceps_help": "Pazu çevresi (cm)",
        "forearm": "Önkol Çevresi (cm)",
        "forearm_help": "Önkol çevresi (cm)",
        "wrist": "Bilek Çevresi (cm)",
        "wrist_help": "Bilek çevresi (cm)",
        "button": "Vücut Yağ Oranını Tahmin Et",
        "success": "Tahmin Tamamlandı!",
        "metric_label": "Tahmin Edilen Vücut Yağ Oranı",
    },
    "English": {
        "title": "🏋️‍♂️ Body Fat Percentage Calculator with AI",
        "description": "Please enter your anatomical measurements below.",
        "age": "Age",
        "age_help": "Your age (years)",
        "weight": "Weight (kg)",
        "weight_help": "Your weight (kg)",
        "height": "Height (cm)",
        "height_help": "Your height (cm)",
        "neck": "Neck Circumference (cm)",
        "neck_help": "Neck circumference (cm)",
        "chest": "Chest Circumference (cm)",
        "chest_help": "Chest circumference (cm)",
        "abdomen": "Abdomen Circumference (cm)",
        "abdomen_help": "Abdomen circumference (cm) (at navel level)",
        "hip": "Hip Circumference (cm)",
        "hip_help": "Hip circumference (cm)",
        "thigh": "Thigh Circumference (cm)",
        "thigh_help": "Thigh circumference (cm)",
        "knee": "Knee Circumference (cm)",
        "knee_help": "Knee circumference (cm)",
        "ankle": "Ankle Circumference (cm)",
        "ankle_help": "Ankle circumference (cm)",
        "biceps": "Biceps Circumference (cm)",
        "biceps_help": "Biceps circumference (cm)",
        "forearm": "Forearm Circumference (cm)",
        "forearm_help": "Forearm circumference (cm)",
        "wrist": "Wrist Circumference (cm)",
        "wrist_help": "Wrist circumference (cm)",
        "button": "Predict Body Fat Percentage",
        "success": "Prediction Complete!",
        "metric_label": "Predicted Body Fat Percentage",
    }
}

# Seçili dili al
t = translations[language]

# --- STREAMLIT ARAYÜZÜ / INTERFACE ---
st.title(t["title"])
st.write(t["description"])

st.divider()

# Kullanıcı Giriş Alanları (Tüm 13 özellik)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(t["age"], min_value=15, max_value=100, value=30, help=t["age_help"])
    weight = st.number_input(t["weight"], min_value=40.0, max_value=250.0, value=80.0, step=0.1, help=t["weight_help"])
    height = st.number_input(t["height"], min_value=130.0, max_value=250.0, value=175.0, step=0.1, help=t["height_help"])
    weight=weight * 2.20462
    height = height / 2.54
    neck = st.number_input(t["neck"], min_value=28.0, max_value=50.0, value=38.0, step=0.1, help=t["neck_help"])
    chest = st.number_input(t["chest"], min_value=80.0, max_value=130.0, value=100.0, step=0.1, help=t["chest_help"])

with col2:
    abdomen = st.number_input(t["abdomen"], min_value=70.0, max_value=140.0, value=90.0, step=0.1, help=t["abdomen_help"])
    hip = st.number_input(t["hip"], min_value=80.0, max_value=130.0, value=100.0, step=0.1, help=t["hip_help"])
    thigh = st.number_input(t["thigh"], min_value=40.0, max_value=70.0, value=55.0, step=0.1, help=t["thigh_help"])
    knee = st.number_input(t["knee"], min_value=30.0, max_value=45.0, value=37.0, step=0.1, help=t["knee_help"])

with col3:
    ankle = st.number_input(t["ankle"], min_value=18.0, max_value=28.0, value=22.0, step=0.1, help=t["ankle_help"])
    biceps = st.number_input(t["biceps"], min_value=25.0, max_value=45.0, value=32.0, step=0.1, help=t["biceps_help"])
    forearm = st.number_input(t["forearm"], min_value=20.0, max_value=35.0, value=28.0, step=0.1, help=t["forearm_help"])
    wrist = st.number_input(t["wrist"], min_value=15.0, max_value=20.0, value=17.0, step=0.1, help=t["wrist_help"])

st.divider()

# --- TAHMİN VE HESAPLAMA MOTORU ---
if st.button(t["button"], type="primary"):
    # Modelin beklediği girdi formatına göre verileri düzenle
    # Özelliklerin sırası, model eğitilirken kullanılan X DataFrame'indeki sütun sırasıyla aynı olmalıdır.
    input_data = pd.DataFrame([{
        'Age': age,
        'Weight': weight,
        'Height': height,
        'Neck': neck,
        'Chest': chest,
        'Abdomen': abdomen,
        'Hip': hip,
        'Thigh': thigh,
        'Knee': knee,
        'Ankle': ankle,
        'Biceps': biceps,
        'Forearm': forearm,
        'Wrist': wrist
    }])

    # Tahmini yap
    predicted_body_fat = model.predict(input_data)[0]

    # Negatif veya mantıksız uç değer çıkma ihtimaline karşı sınırlandırma
    predicted_body_fat = max(2.0, min(predicted_body_fat, 50.0))

    # Sonuçları Ekrana Basma
    st.success(t["success"])

    st.metric(label=t["metric_label"], value=f"% {predicted_body_fat:.2f}")
