
import streamlit as st
import joblib
import numpy as np
import pandas as pd # Import pandas to create a DataFrame for prediction

# 1. Eğitilmiş Modeli Yükle
@st.cache_resource
def load_model():
    # Kendi kaydettiğin model dosyasının adını buraya yazmalısın
    return joblib.load("vucut_yag_modeli.pkl")

model = load_model()

# --- STREAMLIT ARAYÜZÜ ---
st.title("🏋️‍♂️ Yapay Zeka ile Vücut Yağ Oranı Hesaplayıcı")
st.write("Lütfen aşağıdaki anatomik ölçülerinizi giriniz.")

st.divider()

# Kullanıcı Giriş Alanları (Tüm 13 özellik)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Yaş", min_value=15, max_value=100, value=30, help="Yaşınız (yıl)")
    weight = st.number_input("Kilo (lbs)", min_value=100.0, max_value=300.0, value=170.0, step=0.1, help="Kilonuz (libre)")
    height = st.number_input("Boy (inç)", min_value=60.0, max_value=80.0, value=70.0, step=0.1, help="Boyunuz (inç)")
    neck = st.number_input("Boyun Çevresi (cm)", min_value=30.0, max_value=50.0, value=38.0, step=0.1, help="Boyun çevresi (cm)")
    chest = st.number_input("Göğüs Çevresi (cm)", min_value=80.0, max_value=130.0, value=100.0, step=0.1, help="Göğüs çevresi (cm)")

with col2:
    abdomen = st.number_input("Karın Çevresi (cm)", min_value=70.0, max_value=140.0, value=90.0, step=0.1, help="Karın çevresi (cm) (göbek hizasından)")
    hip = st.number_input("Kalça Çevresi (cm)", min_value=80.0, max_value=130.0, value=100.0, step=0.1, help="Kalça çevresi (cm)")
    thigh = st.number_input("Uyluk Çevresi (cm)", min_value=40.0, max_value=70.0, value=55.0, step=0.1, help="Uyluk çevresi (cm)")
    knee = st.number_input("Diz Çevresi (cm)", min_value=30.0, max_value=45.0, value=37.0, step=0.1, help="Diz çevresi (cm)")

with col3:
    ankle = st.number_input("Ayak Bileği Çevresi (cm)", min_value=18.0, max_value=28.0, value=22.0, step=0.1, help="Ayak bileği çevresi (cm)")
    biceps = st.number_input("Pazu Çevresi (cm)", min_value=25.0, max_value=45.0, value=32.0, step=0.1, help="Pazu çevresi (cm)")
    forearm = st.number_input("Önkol Çevresi (cm)", min_value=20.0, max_value=35.0, value=28.0, step=0.1, help="Önkol çevresi (cm)")
    wrist = st.number_input("Bilek Çevresi (cm)", min_value=15.0, max_value=20.0, value=17.0, step=0.1, help="Bilek çevresi (cm)")

st.divider()

# --- TAHMİN VE HESAPLAMA MOTORU ---
if st.button("Vücut Yağ Oranını Tahmin Et", type="primary"):
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
    st.success("Tahmin Tamamlandı!")

    st.metric(label="Tahmin Edilen Vücut Yağ Oranı", value=f"% {predicted_body_fat:.2f}")
