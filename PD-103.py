import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import os
import warnings
import altair as alt
from sklearn.utils.validation import joblib

st.set_page_config(page_title="Body Performance", page_icon='icon.png')

st.title("UAS PENAMBANGAN DATA")

description, importdata, implementation = st.tabs(["Deskripsi", "Import Data", "Implementation"])
# warnings.filterwarnings("ignore")
with description:
    st.subheader("Deskripsi")
    st.write("Nama : Abdul Wachid Al Aziz | NIM : 200411100103 | Kelas : Penambangan Data A")
    st.write("")
    st.write("Dataset berisi tentang catatan performa tubuh manusia, kekuatan fisik dan nilai demografi seperti usia.")
    st.write("Aplikasi ini digunakan untuk mencari klasifikasi performa tubuh.")
    st.write("Fitur yang digunakan :")
    st.write("1. age (Usia) : Numerik")
    st.write("2. gender (Jenis kelamin) : Kategorikal")
    st.write("3. height_cm (Tinggi Badan) : Numerik")
    st.write("4. weight_kg (Berat badan) : Numerik")
    st.write("5. body_fat (Lemak tubuh) : Numerik")
    st.write("6. diastolic (Tekanan darah diastolic) : Numerik")
    st.write("7. systolic (Tekanan darah systolic) : Numerik")
    st.write("8. gripForce (Kekuatan cengkrama) : Numerik")
    st.write("9. sit_and_binding_fowrard (Jarak sitting forward binding : Numerik")
    st.write("10. sit_ups_count (Jumlah sit up) : Numerik")
    st.write("11. broad_jumps_cm (Nilai lompat jauh  : Numerik")
    st.write("Sumber dataset https://www.kaggle.com/datasets/kukuroo3/body-performance-data")
    st.write("Link github https://github.com/alaziz31/interface")

with importdata:
    st.subheader("Upload File .csv")
    uploaded_files = st.file_uploader("", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        dataset, preprocessing, modelling = st.tabs(["Dataset", "Preprocessing", "Modelling"])
        with dataset:
            df = pd.read_csv(uploaded_file)
            df['gender'].replace(['F','M'],[0,1],inplace=True)
            st.write('keterangan : "gender" 1 = Male, 0 = Female')
            st.dataframe(df)

        with preprocessing:
            st.subheader("Preprocessing")
            prepros = st.radio(
            "Silahkan pilih metode yang digunakan :",
            (["Min Max Scaler"]))
            prepoc = st.button("Preprocessing")

            if prepros == "Min Max Scaler":
                if prepoc:
                    df[["age","gender", "height_cm", "weight_kg", "body_fat", "diastolic", "systolic", "gripForce", "sit_and_bend_forward_cm", "sit_ups_counts","broad_jump_cm"]].agg(['min','max'])
                    df.Class.value_counts()
                    X = df.drop(columns=["Class"],axis=1)
                    y = df["Class"]

                    "### Normalize data transformasi"
                    X
                    X.shape, y.shape
                    # le.inverse_transform(y)
                    labels = pd.get_dummies(df.Class).columns.values.tolist()
                    "### Label"
                    labels
                    """## Normalisasi MinMax Scaler"""
                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    X
                    X.shape, y.shape

        with modelling:
            X=df[["age", "gender", "height_cm", "weight_kg", "body_fat", "diastolic", "systolic", "gripForce", "sit_and_bend_forward_cm", "sit_ups_counts","broad_jump_cm"]]
            y=df["Class"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            # from sklearn.feature_extraction.text import CountVectorizer
            # cv = CountVectorizer()
            # X_train = cv.fit_transform(X_train)
            # X_test = cv.fit_transform(X_test)
            st.subheader("Modeling")
            st.write("Silahkan pilih Model yang ingin anda Modelling :")
            naive = st.checkbox('Naive Bayes')
            kn = st.checkbox('K-Nearest Neighbor')
            des = st.checkbox('Decision Tree')
            mod = st.button("Modeling")

            # NB
            GaussianNB(priors=None)

            # Fitting Naive Bayes Classification to the Training set with linear kernel
            nvklasifikasi = GaussianNB()
            nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = nvklasifikasi.predict(X_test)
            
            y_compare = np.vstack((y_test,y_pred)).T
            nvklasifikasi.predict_proba(X_test)
            akurasi_nb = round(100 * accuracy_score(y_test, y_pred))
            # akurasi_nb = 10

            # KNN 
            K=10
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_train,y_train)
            y_pred=knn.predict(X_test)

            akurasi_knn = round(100 * accuracy_score(y_test,y_pred))

            # DT

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            # prediction
            dt.score(X_test, y_test)
            y_pred = dt.predict(X_test)
            #Accuracy
            akurasi_dt = round(100 * accuracy_score(y_test,y_pred))

            if naive :
                if mod :
                    st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi_nb))
            if kn :
                if mod:
                    st.write("Model KNN accuracy score : {0:0.2f}" . format(akurasi_knn))
            if des :
                if mod :
                    st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasi_dt))


    
with implementation:
    st.subheader("Implementation")
    age = st.number_input('Masukkan Usia (tahun)')
    gender = st.number_input('Masukkan gender ( 1=Pria 0=Wanita )')
    height_cm = st.number_input('Masukkan Tinggi badan (cm)')
    weight_kg = st.number_input('Masukkan Berat badan (Kg)')
    body_fat = st.number_input('Masukkan Lemak tubuh (Persen)')
    diastolic = st.number_input('Masukkan Tekanan diastolic')
    systolic = st.number_input('Masukkan Tekanan systolic')
    gripForce = st.number_input('Masukkan Kekuatan cengkraman (Kg)')
    sit_and_bend_forward_cm = st.number_input('Masukkan Ukuran sit and bend (cm)')
    sit_ups_counts = st.number_input('Masukkan Jumlah sit-ups')
    broad_jump_cm = st.number_input('Masukkan Jarak broad jump (cm)')

    def submit():
        # input
        inputs = np.array([[
            age, height_cm, weight_kg, body_fat, diastolic, systolic, gripForce, sit_and_bend_forward_cm, sit_ups_counts, broad_jump_cm
        ]])
        baru = pd.DataFrame(inputs)
        input = pd.get_dummies(baru)
        st.write("Data yang diinputkan :")
        st.write(input)
        inputan = np.array(input)
        le = joblib.load("le.save")
        model1 = joblib.load("tree.joblib")
        y_pred3 = model1.predict(inputs)
        st.write("Berdasarkan data yang diinputkan, didapatkan Body Performance dengan Grade : ", le.inverse_transform(y_pred3)[0])

    all = st.button("Submit")
    if all :
        st.snow()
        submit()
