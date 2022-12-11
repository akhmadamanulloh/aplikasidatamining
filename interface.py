import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.write(""" # APLIKASI PREDIKSI GAGAL JANTUNG """)
st.write("Oleh: Mochammad Rizki Aji Santoso (200411100086)")
st.write("----------------------------------------------------------------------------------")

deskripsi, datasets, preprocessing, modeling, implementasi = st.tabs(["Deskripsi", "Dataset", "Preprocessing", "Modelling", "Implementation"])

with deskripsi:
    st.subheader("""Tentang Dataset""")
    st.write("Penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global , merenggut sekitar 17,9 juta nyawa setiap tahun , yang merupakan 31% dari semua kematian di seluruh dunia . Gagal jantung adalah kejadian umum yang disebabkan oleh CVD dan kumpulan data ini berisi 12 fitur yang dapat digunakan untuk memprediksi kematian akibat gagal jantung.")
    st.write("Sebagian besar penyakit kardiovaskular dapat dicegah dengan mengatasi faktor risiko perilaku seperti penggunaan tembakau, pola makan yang tidak sehat dan obesitas, kurangnya aktivitas fisik, dan penggunaan alkohol yang berbahaya dengan menggunakan strategi populasi luas.")
    st.write("Orang dengan penyakit kardiovaskular atau yang memiliki risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau penyakit yang sudah ada) memerlukan deteksi dan penanganan dini di mana model pembelajaran mesin dapat sangat membantu.")
    st.write("""Dataset gagal jantung ini diambil dari <a href="https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data">Kaggle</a>""", unsafe_allow_html=True)
    st.subheader("""Fitur""")
    st.write(
        """
        Fitur yang terdapat pada dataset:
        - Age : Usia pasien
        - Anaemia : Masalah kesehatan yang terjadi saat jumlah sel darah merah dalam tubuh lebih rendah dibandingkan dengan jumlah normalnya, sering dikenal dengan penyakit kekurangan sel darah merah
        - Creatinine Phosphokinase: Creatine Phosphokinase (CK) adalah sejenis protein yang dikenal sebagai enzim. Protein tersebut sebagian besar ditemukan di otot rangka dan jantung, dengan jumlah yang lebih sedikit di otak. Sedangkan tes CK digunakan untuk mengukur jumlah creatine kinase dalam darah. 
        - Diabetes: Diabetes atau penyakit gula adalah penyakit kronis atau yang berlangsung jangka panjang. Penyakit ini ditandai dengan meningkatnya kadar gula darah (glukosa) hingga di atas nilai normal.
        - Ejection Fraction: Dalam pemeriksaan USG jantung, yang paling penting dinilai ialah nilai fraksi ejeksi / ejection fraction (EF). Fraksi ejeksi mencerminkan seberapa banyak darah yang terpompa dibandingkan dengan jumlah darah yang masuk ke dalam kamar jantung. Normalnya, nilai EF berkisar antara 50-70%.
        - High Blood Pressure: Tekanan darah tinggi atau disebut juga hipertensi adalah suatu kondisi ketika seseorang mempunyai tekanan darah yang terukur pada nilai 130/80 mmHg atau lebih tinggi.
        - Platelets: Trombosit (keping darah/platelet) adalah komponen darah yang berfungsi dalam pembekuan darah. Jumlahnya yang terlalu rendah dapat membuat Anda mudah memar dan mengalami perdarahan.
        - Serum Creatinine: Serum kreatinin merupakan sampah hasil metabolisme otot yang mengalir pada sirkulasi darah. Kreatinin lalu disaring ginjal untuk selanjutnya dibuang bersama urine. Serum kreatinin menjadi pertanda baik buruknya fungsi ginjal, karena organ ini yang mengatur agar kreatinin tetap berada pada kadar normalnya.
        - Serum Sodium: Kadar natrium serum adalah parameter utama yang digunakan untuk menilai tonisitas serum yang sering terganggu akibat hiperglikemia. Efek hiperglikemia terhadap penurunan konsentrasi natrium plasma telah diketahui sejak separuh abad yang lalu.
        - Sex: Jenis Kelamin 
        """
    )

with datasets:
    st.subheader("""Dataset Heart Failure Prediction""")
    df = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/main/main/tennis.csv')
    st.dataframe(df) 

with preprocessing:
    st.subheader("""Rumus Normalisasi Data""")
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    df_dum=pd.get_dummies(data=df,columns=['temp','outlook','humidity','windy'])
    df_dum
    X = df_dum.drop(columns=['play'])
    y = df_dum['play'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.play).columns.values.tolist()
    dumies 
