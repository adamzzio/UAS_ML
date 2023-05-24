# ===== IMPORT LIBRARY =====
# for data wrangling
import pandas as pd
import numpy as np

# for web app
import streamlit as st
from streamlit_lottie import st_lottie

# for modelling
import pickle as pkl
from lightgbm import LGBMClassifier

# for database
import firebase_admin
import csv
import google.cloud
from firebase_admin import credentials, firestore
from google.cloud import firestore

# etc
from PIL import Image
import requests

# ===== SET PAGE =====
pageicon = Image.open("aset_foto/CardioCheck.png")
st.set_page_config(page_title="CardioCheck Web App", page_icon=pageicon, layout="wide")

# ===== INITIALIZE DATABASE CONNECTION =====
# # Inisialisasi Firebase Admin SDK
# def initialize_firebase():
#     cred = credentials.Certificate("uas-ml-3772c-firebase-adminsdk-x4nhg-b013b10236.json")
#     firebase_admin.initialize_app(cred)
    
# def save_data_to_firebase(data):
#     db = firestore.client()
#     collection_name = "dataset_ML"
#     doc_ref = db.collection(collection_name).document()
#     doc_ref.set(data)

# Buat fungsi baru untuk mendapatkan referensi ke Firebase App
def get_firebase_app():
    # Periksa apakah aplikasi Firebase sudah ada
    if not firebase_admin._apps:
        # Jika belum ada, inisialisasi Firebase Admin SDK
        cred = credentials.Certificate("uas-ml-3772c-firebase-adminsdk-x4nhg-b013b10236.json")
        firebase_admin.initialize_app(cred)
    # Kembalikan referensi ke aplikasi Firebase
    return firebase_admin.get_app()

# Ganti pemanggilan fungsi save_data_to_firebase dengan menggunakan get_firebase_app()
def save_data_to_firebase(data):
    app = get_firebase_app()
    db = firestore.client(app)
    collection_name = "dataset_ML"
    doc_ref = db.collection(collection_name).document()
    doc_ref.set(data)
    
def save_data_to_firebase_feedback(feedback):
    app = get_firebase_app()
    db = firestore.client(app)
    collection_name = "feedback"
    doc_ref = db.collection(collection_name).document()
    doc_ref.set(feedback)

def save_data_to_db(data, feedback):
    app = get_firebase_app()
    db = firestore.client(app)
    collection_name_data = "dataset_ML"
    collection_name_feed = 'feedback'
    doc_ref_data = db.collection(collection_name_data).document()
    doc_ref_feed = db.collection(collection_name_feed).document()
    doc_ref_data.set(data)
    doc_ref_feed.set(feedback)
    
# ===== LOAD MODEL & DATA =====

filename_model = 'finalized_model_lgbm_tuning.sav'

@st.cache_resource
def load_model():
    model = pkl.load(open(filename_model, 'rb'))
    return model

model = load_model()

# ===== DEVELOP FRONT-END =====
# SET HEADER PAGE
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_uwWgICKCxj.json")

intro_column_left, intro_column_right = st.columns(2)
with st.container():
    with intro_column_left:
        # st.title(":bar_chart: Dashboard")
        st.markdown('<div style="text-align: justify; font-size:300%; line-height: 150%; margin-top: -55px;"> <b><br>CardioCheck: Your Reliable Cardiovascular Decision Support </b> </div>',
            unsafe_allow_html=True)
    with intro_column_right:
        st_lottie(lottie_coding, height=250, key="dashboard")

st.markdown('<hr>', unsafe_allow_html=True)

# SET DESCRIPTION
st.markdown('<div style="text-align: justify; font-size:160%; text-indent: 4em;"> CardioCheck adalah sebuah aplikasi web yang bertujuan menjadi alat bantu pengambil keputusan yang handal dalam bidang kesehatan kardiovaskular. Dengan visi "Empowering Informed Decisions" CardioCheck dirancang untuk memberikan informasi yang akurat dan cepat kepada para tenaga ahli terkait, seperti dokter, dalam proses pengambilan keputusan terkait kondisi kardiovaskular.</div>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align: justify; font-size:160%; text-indent: 4em;"> Fungsi utama CardioCheck adalah melakukan pengecekan dan evaluasi kondisi kardiovaskular berdasarkan data pasien. Dengan menggunakan model Machine Learning yang terlatih, aplikasi ini memberikan prediksi risiko penyakit jantung berdasarkan faktor-faktor seperti usia, jenis kelamin, tekanan darah, kolesterol, dan variabel lainnya.</div>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align: justify; font-size:160%; text-indent: 4em;"> Kami adalah tim yang berdedikasi di balik CardioCheck, terdiri dari para ahli kesehatan dan ilmu data yang memiliki komitmen kuat terhadap pengembangan solusi inovatif dalam bidang kardiovaskular. Dengan pengetahuan medis yang mendalam dan keahlian dalam analisis data, kami berupaya memberikan alat yang dapat diandalkan bagi para tenaga ahli terkait dalam mendukung pengambilan keputusan yang lebih baik.</div>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align: justify; font-size:160%; text-indent: 4em;"> Dengan CardioCheck, kami berharap dapat memberikan solusi yang memberdayakan para dokter dalam membuat keputusan yang informasional dan terarah dalam hal kardiovaskular, sehingga meningkatkan kualitas perawatan pasien dan memberikan manfaat yang signifikan bagi komunitas medis. </div>',
            unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# CREATE FORM
age = st.number_input(label = 'Masukkan usia pasien : ', min_value=18, max_value=80, step=1, key='1')
gender = st.selectbox('Masukkan jenis kelamin pasien', ('Laki-Laki', 'Perempuan'))
heart_rate = st.number_input(label = 'Masukkan detak jantung pasien : ', min_value=0, max_value=200, step=10, key='2')
systolic = st.number_input(label = 'Masukkan tekanan sistolik pasien : ', min_value=0, max_value=200, step=10, key='3')
diastolic = st.number_input(label = 'Masukkan tekanan diastolik pasien : ', min_value=0, max_value=200, step=10, key='4')
blood_sugar = st.number_input(label = 'Masukkan kadar gula darah pasien : ', min_value=0, max_value=1000, step=10, key='5')
ckmb = st.number_input(label = 'Masukkan kadar CK-Mb pasien : ', min_value=0, max_value=100, step=10, key='6')
troponin = st.number_input(label = 'Masukkan kadar troponin pasien : ', min_value=0.0, max_value=100.0, step=0.1, key='7')

submit = st.button("Submit", use_container_width=True)

# ===== BACK-END SESSIONS ======
# SAVE RESULT TO DATAFRAME
df_result = pd.DataFrame({'Age':[age],
                          'Gender':[gender],
                          'Heart rate':[heart_rate],
                          'Systolic blood pressure':[systolic],
                          'Diastolic blood pressure':[diastolic],
                          'Blood sugar':[blood_sugar],
                          'CK-MB':[ckmb],
                          'Troponin':[troponin]})

gender_dict = {'Laki-Laki':1,
               'Perempuan':0}
df_result['Gender'] = df_result['Gender'].map(gender_dict)

# DO PREDICTIONS
if submit:
    result = model.predict(df_result.values)
    result_proba = model.predict_proba(df_result.values)
    # result_proba = str(result_proba)
    result_proba = np.max(result_proba[0])
    result_proba = np.round(result_proba, 2)
    result_proba = result_proba * 100
    if result == 0:
        text_result = "Pasien Anda memiliki peluang " + str(result_proba) + "% dinyatakan negatif memiliki penyakit jantung"
        st.success(text_result)
        st.balloons()
        # SUBMIT PREDICTIONS TO DATABASE
        df_result['Result'] = 'negative'
        data = {'Age':age,
                'Gender':gender,
                'Heart rate':heart_rate,
                'Systolic blood pressure':systolic,
                'Diastolic blood pressure':diastolic,
                'Blood sugar':blood_sugar,
                'CK-MB':ckmb,
                'Troponin':troponin,
                'Result':'negative'}
        save_data_to_firebase(data)
        st.success("Data Anda berhasil disimpan ke database")
        st.markdown('<hr>', unsafe_allow_html=True)
        option = st.selectbox(
            'Bagaimana perasaan Anda setelah menggunakan Web App ini?',
            ('Puas', 'Tidak Puas'))
        submit_feed = st.button("Submit Feedback", use_container_width=True)
        if submit_feed:
            feed = {"kepuasan":option}
            save_data_to_firebase_feedback()
            st.success("Feedback Anda berhasil disimpan ke database")
        
#         st.markdown("<h1 style='text-align: center; color: white;'>Apakah Anda puas? </h1>", unsafe_allow_html=True)
#         img_left, img_right = st.columns(2)
#         with img_left:
#             st.image(Image.open("aset_foto/aset_baiklah.jpg"), use_column_width=True)
#         with img_right:
#             st.image(Image.open("aset_foto/aset_gabahaya.jpg"), use_column_width=True)

#         feed_left, feed_right = st.columns(2)
#         with feed_left:
#             puas = st.button('Puas', use_container_width=True)
#             if puas:
#                 save_data_to_db(data, {'kepuasan':'Puas'})
#                 st.success("Feedback Anda berhasil disimpan ke database")
#         with feed_right:
#             tdk_puas = st.button('Tidak Puas', use_container_width=True)
#             if tdk_puas:
#                 save_data_to_db(data, {'kepuasan':'Tidak Puas'})
#                 st.success("Feedback Anda berhasil disimpan ke database")
        
    else:
        text_result = "Pasien Anda memiliki peluang " + str(result_proba) + "% dinyatakan positif memiliki penyakit jantung"
        st.error(text_result)
        st.balloons()
        # SUBMIT PREDICTIONS TO DATABASE
        df_result['Result'] = 'positive'
        data = {'Age':age,
                'Gender':gender,
                'Heart rate':heart_rate,
                'Systolic blood pressure':systolic,
                'Diastolic blood pressure':diastolic,
                'Blood sugar':blood_sugar,
                'CK-MB':ckmb,
                'Troponin':troponin,
                'Result':'positive'}
        save_data_to_firebase(data)
        st.success("Data Anda berhasil disimpan ke database")
        st.markdown('<hr>', unsafe_allow_html=True)
        option = st.selectbox(
            'Bagaimana perasaan Anda setelah menggunakan Web App ini?',
            ('Puas', 'Tidak Puas'))
        submit_feed = st.button("Submit Feedback", use_container_width=True)
        if submit_feed:
            feed = {"kepuasan":option}
            save_data_to_firebase_feedback()
            st.success("Feedback Anda berhasil disimpan ke database")
#         text_result = "Pasien Anda memiliki peluang " + str(result_proba) + "% dinyatakan positif memiliki penyakit jantung"
#         st.error(text_result)
#         st.balloons()
#         # SUBMIT PREDICTIONS TO DATABASE
#         df_result['Result'] = 'positive'
#         data = {'Age':age,
#                 'Gender':gender,
#                 'Heart rate':heart_rate,
#                 'Systolic blood pressure':systolic,
#                 'Diastolic blood pressure':diastolic,
#                 'Blood sugar':blood_sugar,
#                 'CK-MB':ckmb,
#                 'Troponin':troponin,
#                 'Result':'positive'}
        
#         st.markdown("<h1 style='text-align: center; color: white;'>Apakah Anda puas? </h1>", unsafe_allow_html=True)
#         img_left, img_right = st.columns(2)
#         with img_left:
#             st.image(Image.open("aset_foto/aset_baiklah.jpg"), use_column_width=True)
#         with img_right:
#             st.image(Image.open("aset_foto/aset_gabahaya.jpg"), use_column_width=True)

#         feed_left, feed_right = st.columns(2)
#         with feed_left:
#             puas = st.button('Puas', use_container_width=True)
#             if puas:
#                 save_data_to_db(data, {'kepuasan':'Puas'})
#                 st.success("Feedback Anda berhasil disimpan ke database")
#         with feed_right:
#             tdk_puas = st.button('Tidak Puas', use_container_width=True)
#             if tdk_puas:
#                 save_data_to_db(data, {'kepuasan':'Tidak Puas'})
#                 st.success("Feedback Anda berhasil disimpan ke database")

#     st.markdown('<hr>', unsafe_allow_html=True)
    # FEEDBACK SESSIONS
    # _, mid_feed, _ = st.columns([1,6,1])
    # with mid_feed:
    #     st.write('## Apakah Anda puas?')
    # st.title('Apakah Anda puas?')
#     st.markdown("<h1 style='text-align: center; color: white;'>Apakah Anda puas? </h1>", unsafe_allow_html=True)

#     img_left, img_right = st.columns(2)
#     with img_left:
#         st.image(Image.open("aset_foto/aset_baiklah.jpg"), use_column_width=True)
#     with img_right:
#         st.image(Image.open("aset_foto/aset_gabahaya.jpg"), use_column_width=True)

#     feed_left, feed_right = st.columns(2)
#     with feed_left:
#         st.button('Puas', use_container_width=True)
#     with feed_right:
#         st.button('Tidak Puas', use_container_width=True)
