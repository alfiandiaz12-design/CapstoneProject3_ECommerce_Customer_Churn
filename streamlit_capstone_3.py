"""
Streamlit demo — Prediksi Customer Churn
Model: Soft Voting (RandomForest + XGBoost + Logistic Regression)

Cara pakai:
1. Taruh file 'soft_final_model2.pkl' di folder yang sama dengan app.py
2. Jalankan: streamlit run app.py
"""

import pickle
import pandas as pd
import streamlit as st

# ── 1. Konstanta ──────────────────────────────────────────────────────────
MODEL_PATH = "soft_final_model2.pkl"
THRESHOLD = 0.47  # threshold hasil tuning di notebook, tervalidasi di holdout test (recall 0.85, gap 0.18, akurasi 0.90)
KATEGORI_LEMAH = ["Grocery", "Others", "Mobile Phone"]

# ── 2. Load model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"⚠️ File model '{MODEL_PATH}' tidak ditemukan di folder yang sama! Harap unggah file model terlebih dahulu.")
    st.stop()

# ── 3. Judul halaman ──────────────────────────────────────────────────────
st.title("📉 Prediksi Customer Churn")
st.caption("Model: Soft Voting (RandomForest + XGBoost + Logistic Regression)")

# ── 4. Form input (satu per satu, urut dari atas ke bawah) ────────────────
tenure = st.number_input("Tenure (bulan berlangganan)", 0, 100, 10)
warehouse_to_home = st.number_input("Warehouse To Home (jarak, km)", 0, 200, 15)
number_of_device = st.number_input("Number Of Device Registered", 1, 10, 3)
prefered_order_cat = st.selectbox(
    "Prefered Order Category",
    ["Fashion", "Grocery", "Laptop & Accessory", "Mobile Phone", "Others"],
)
satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_address = st.number_input("Number Of Address", 1, 20, 2)
complain = st.radio("Pernah Komplain?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
day_since_last_order = st.number_input("Day Since Last Order", 0, 100, 5)
cashback_amount = st.number_input("Cashback Amount", 0.0, 1000.0, 150.0)

# ── 5. Tombol prediksi ─────────────────────────────────────────────────────
if st.button("Prediksi Churn"):

    # 5a. Susun input jadi satu baris data
    data = {
        "Tenure": tenure,
        "WarehouseToHome": warehouse_to_home,
        "NumberOfDeviceRegistered": number_of_device,
        "PreferedOrderCat": prefered_order_cat,
        "SatisfactionScore": satisfaction_score,
        "MaritalStatus": marital_status,
        "NumberOfAddress": number_of_address,
        "Complain": complain,
        "DaySinceLastOrder": day_since_last_order,
        "CashbackAmount": cashback_amount,
    }

    # 5b. Feature engineering — harus sama persis dengan notebook training
    data["Tenure_Complain_Interaction"] = data["Tenure"] * data["Complain"]
    data["Cashback_per_Tenure"] = data["CashbackAmount"] / (data["Tenure"] + 1)
    if data["PreferedOrderCat"] in KATEGORI_LEMAH:
        data["PreferedOrderCat"] = "Other_Product"

    X_input = pd.DataFrame([data])

    # 5c. Prediksi — pakai threshold hasil tuning, bukan default 0.5
    try:
        proba_churn = model.predict_proba(X_input)[0, 1]
        prediction = 1 if proba_churn >= THRESHOLD else 0

        # 5d. Tampilkan hasil
        st.write("---")
        st.metric("Probabilitas Churn", f"{proba_churn:.1%}")

        if prediction == 1:
            st.error(f"⚠️ Pelanggan ini diprediksi **berpotensi churn**.")
        else:
            st.success(f"✅ Pelanggan ini diprediksi **tetap aktif**.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi dengan model: {e}")