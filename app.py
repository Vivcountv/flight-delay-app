import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import datetime

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Keterlambatan Penerbangan", layout="wide")
st.title('✈️ Prediktor Keterlambatan Penerbangan')
st.write("Aplikasi ini memprediksi apakah penerbangan Anda akan terlambat lebih dari 15 menit menggunakan LightGBM.")

# --- 2. Muat Model dan Data Pendukung ---
# Gunakan cache agar model tidak dimuat ulang setiap kali
@st.cache_resource
def load_assets():
    model = joblib.load('flight_delay_model.joblib')
    weather_daily = pd.read_csv('weather_daily_processed.csv')
    model_columns = joblib.load('model_columns.joblib')
    categorical_cols = joblib.load('categorical_features.joblib')
    
    # Konversi kolom tanggal di data cuaca saat dimuat
    weather_daily['merge_key_date'] = pd.to_datetime(weather_daily['merge_key_date']).dt.date
    return model, weather_daily, model_columns, categorical_cols

model, weather_daily, model_columns, categorical_cols = load_assets()

st.subheader("Masukkan Detail Penerbangan:")

# --- 3. Buat Form Input (UI) ---
# Kita gunakan fitur-fitur paling penting dari analisis Anda
col1, col2 = st.columns(2)

with col1:
    # Input Tanggal - Penting untuk merge cuaca
    flight_date_input = st.date_input("Tanggal Penerbangan", datetime.date(2023, 1, 15))
    
    # Input Airline - Fitur kategorikal
    # (Ini hanya contoh, Anda bisa ambil daftar lengkap dari df_final['Airline'].unique())
    airline_input = st.selectbox("Maskapai (Airline)", 
                                 ['Endeavor Air', 'Frontier Airlines Inc.', 'JetBlue Airways', 
                                  'Republic Airways', 'Southwest Airlines Co.', 'Delta Air Lines Inc.'])
    
    # Input Delay Keberangkatan - Fitur Numerik Terpenting
    dep_delay_input = st.number_input("Keterlambatan Keberangkatan (Dep_Delay)", 
                                      min_value=-60, max_value=300, value=0,
                                      help="Masukkan 0 jika tepat waktu, atau angka negatif jika berangkat lebih awal.")

with col2:
    # Input Bandara Asal & Tujuan - Penting untuk merge cuaca
    # (Ini hanya contoh, Anda bisa ambil daftar lengkap dari df_final['Dep_Airport'].unique())
    dep_airport_input = st.selectbox("Bandara Keberangkatan (Dep_Airport)", 
                                     ['ATL', 'LGA', 'DFW', 'ORD', 'DEN', 'LAX'])
    
    arr_airport_input = st.selectbox("Bandara Tujuan (Arr_Airport)", 
                                     ['CVG', 'BGM', 'MSP', 'FAY', 'ORD', 'ATL'])

    # Input Durasi - Fitur Numerik Penting
    duration_input = st.number_input("Durasi Penerbangan (Menit)", min_value=30, max_value=600, value=120)


# --- 4. Tombol Prediksi dan Logika Backend ---
if st.button('🚀 Prediksi Keterlambatan'):
    
    # Kunci untuk merge cuaca
    merge_key = flight_date_input
    
    # --- 5. Lakukan Feature Engineering (SAMA SEPERTI DI NOTEBOOK) ---
    
    # 5a. Dapatkan data cuaca asal (origin)
    origin_weather = weather_daily[
        (weather_daily['airport_id'] == dep_airport_input) & 
        (weather_daily['merge_key_date'] == merge_key)
    ].copy()
    
    # Ganti nama kolom origin
    origin_rename = {col: f"origin_{col}" for col in origin_weather.columns if col not in ['airport_id', 'merge_key_date']}
    origin_weather = origin_weather.rename(columns=origin_rename)
    
    # 5b. Dapatkan data cuaca tujuan (destination)
    dest_weather = weather_daily[
        (weather_daily['airport_id'] == arr_airport_input) & 
        (weather_daily['merge_key_date'] == merge_key)
    ].copy()

    # Ganti nama kolom destination
    dest_rename = {col: f"dest_{col}" for col in dest_weather.columns if col not in ['airport_id', 'merge_key_date']}
    dest_weather = dest_weather.rename(columns=dest_rename)

    # --- 6. Siapkan Input DataFrame untuk Model ---
    
    # Buat dictionary untuk data input tunggal
    input_data = {
        'Airline': airline_input,
        'Dep_Airport': dep_airport_input,
        'Arr_Airport': arr_airport_input,
        'Dep_Delay': dep_delay_input,
        'Flight_Duration': duration_input,
        # Tambahkan fitur lain sesuai kebutuhan...
        # Fitur yang tidak ada di input form, kita isi default (cth: 0 atau 'Unknown')
        'Day_Of_Week': flight_date_input.weekday() + 1, # Sesuaikan dengan format data Anda
        'Dep_CityName': 'Unknown',
        'DepTime_label': 'Unknown',
        'Arr_CityName': 'Unknown',
        'Distance_type': 'Unknown',
        'Manufacturer': 'Unknown',
        'Model': 'Unknown',
        'Aicraft_age': 0
    }

    # Buat DataFrame 1 baris
    input_df = pd.DataFrame([input_data])

    # Gabungkan fitur cuaca (reset index agar bisa digabung ke 1 baris)
    input_df = pd.concat([
        input_df.reset_index(drop=True),
        origin_weather.reset_index(drop=True).drop(columns=['airport_id', 'merge_key_date'], errors='ignore'),
        dest_weather.reset_index(drop=True).drop(columns=['airport_id', 'merge_key_date'], errors='ignore')
    ], axis=1)

    # Isi NaN jika data cuaca tidak ditemukan (diisi 0 seperti di notebook)
    input_df = input_df.fillna(0)

    # Pastikan urutan kolom SAMA PERSIS dengan saat pelatihan
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Ubah tipe data kategorikal SAMA PERSIS dengan saat pelatihan
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')

    # --- 7. Lakukan Prediksi ---
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probabilitas delay

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.error(f"**Penerbangan Diprediksi TERLAMBAT** (Probabilitas: {probability*100:.2f}%)")
        else:
            st.success(f"**Penerbangan Diprediksi TEPAT WAKTU** (Probabilitas Terlambat: {probability*100:.2f}%)")
            
        # (Opsional) Tampilkan data yang digunakan untuk prediksi
        with st.expander("Lihat data yang digunakan untuk prediksi (setelah feature engineering)"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        st.error("Ini mungkin terjadi jika data cuaca untuk bandara/tanggal tersebut tidak tersedia di dataset sampel.")