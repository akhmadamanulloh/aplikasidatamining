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

# display
# st.set_page_config(layout='wide')
st.set_page_config(page_title="weather play tennis")

st.title("UAS PENDAT")
st.write("By: Akhmad Amanulloh (20041110099)")
dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data", "Prepocessing", "Modeling", "Implementation"])

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
            dt['outlook'].unique()
            dt_dum=pd.get_dummies(data=dt,columns=['temp','outlook','humidity','windy'])
            dt_dum
    with modeling:
        X = dt_dum.drop('play',axis=1)
        y = dt_dum['play']
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5,random_state=1)
        mlpc, knc, dtc = st.tabs(
        ["MLPClassifier", "KNeighborsClassifier", "DecisionTreeClassifier"])
        with mlpc:
            clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
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
        model = st.selectbox('Pilih Model', ['MLPC', 'KNN', 'DTREE'])
        if model == 'MLPC':
            y_imp = clf.predict(data)
        elif model == 'KNN':
            y_imp = knn.predict(data)
        else:
            y_imp = classifier.predict(data)
        st.success(f'Model yang dipilih = {model}')
        st.success(f'Data Predict = {y_imp}')
