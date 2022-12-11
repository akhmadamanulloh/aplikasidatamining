import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB

# display
# st.set_page_config(layout='wide')
st.set_page_config(page_title="weather play tennis")

st.title("UAS PENDAT")
st.write("By: Akhmad Amanulloh (20041110099)")
deskripsi,dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Deskripsi","Data", "Prepocessing", "Modeling", "Implementation"])
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
with dataframe:
    url = "https://www.kaggle.com/datasets/pranavpandey2511/tennis-weather"
    st.markdown(
        f'[Dataset]({url})')
    st.write('Cuaca yang cocok untuk bermain tennis')

    dt = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/main/main/tennis.csv')
    st.dataframe(dt)
    with preporcessing:
        preporcessingg, ket = st.tabs(['preporcessing', 'Ket preporcessing'])
        with ket:
            st.write("""
                    Keterangan:
                    * 0 : Tidak 
                    * 1 : Iya
                    """)
        with preporcessingg:
            st.subheader("Preprocessing")
            #Mendefinisikan Varible X dan Y
            dt_dum=pd.get_dummies(data=dt,columns=['temp','outlook','humidity','windy'])
            dt_dum
            X = dt_dum.drop(columns=['play'])
            y = dt_dum['play'].values
            dt_min = X.min()
            dt_max = X.max()

            #NORMALISASI NILAI X
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(X)
            features_names = X.columns.copy()
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            st.subheader('Hasil Normalisasi Data')
            st.write(scaled_features)

            st.subheader('Target Label')
            dumies = pd.get_dummies(dt.play).columns.values.tolist()
            dumies 
    with modeling:
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5,random_state=1)
        clf, knc, dtc = st.tabs(
        ["GaussianNB", "KNeighborsClassifier", "DecisionTreeClassifier"])
        with clf:
            clf = GaussianNB(priors=None)
            clf.fit(X_train,y_train)
            y_pred_clf = clf.predict(X_test)
            akurasi_clf = accuracy_score(y_test, y_pred_clf)
            label_clf = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_clf}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_clf}')
            st.dataframe(label_clf)
        with knc:
            knn = KNeighborsClassifier(n_neighbors = 5)
            knn.fit(X_train,y_train)
            y_pred_knn = knn.predict(X_test)
            akurasi_knn = accuracy_score(y_test, y_pred_knn)
            label_knn = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_knn}')
            st.dataframe(label_knn)
        with dtc:
            classifier=DecisionTreeClassifier(criterion='gini')
            classifier.fit(X_train,y_train)
            y_pred_d3 = classifier.predict(X_test)
            akurasi_d3 = accuracy_score(y_test, y_pred_d3)
            label_d3 = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_d3}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_d3}')
            st.dataframe(label_d3)
    with implementation:
        tema = st.selectbox('Temperatur', ['cool', 'hot', 'mild'])
        temp_cool = 1 if tema == 'cool' else 0
        temp_hot = 1 if tema == 'hot' else 0
        temp_mild = 1 if tema == 'mild' else 0

        outlook = st.selectbox('outlook', ['overcast', 'rainy', 'sunny'])
        outlook_overcast = 1 if outlook == 'overcast' else 0
        outlook_rainy = 1 if outlook == 'rainy' else 0
        outlook_sunny = 1 if outlook == 'sunny' else 0

        humidity = st.selectbox('humidity', ['high', 'normal'])
        humidity_high = 1 if humidity == 'high' else 0
        humidity_normal = 1 if humidity == 'normal' else 0

        windy = st.selectbox('windy', ['False', 'True'])
        windy_False = 1 if windy == 'False' else 0
        windy_True = 1 if windy == 'True' else 0

        data = np.array([[temp_cool,temp_hot,temp_mild,outlook_overcast,outlook_rainy,outlook_sunny,humidity_high,humidity_normal,windy_False,windy_True]])
        model = st.selectbox('Pilih Model', ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier'])
        if model == 'GaussianNB':
            y_imp = clf.predict(data)
        elif model == 'KNeighborsClassifier':
            y_imp = knn.predict(data)
        else:
            y_imp = classifier.predict(data)
        st.success(f'Model yang dipilih = {model}')
        st.success(f'Data Predict = {y_imp}')
