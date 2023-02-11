import streamlit as st
import pandas as pd
import preProcessingModul
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
sw=nltk.corpus.stopwords.words("indonesian")

# title and description
st.subheader('#Testing-ProjectModel')
st.write("""
# Deteksi Sarkasme
Uji Coba Hanya Mendukung Bahasa Indonesia 
""")

def prediksi(q):
    DataTweet = pd.DataFrame(data=[q], columns=['Tweet'])
    #============== Start Processing Text
    DataTweet['casefolding'] = DataTweet['Tweet'].apply(preProcessingModul.lower)
    DataTweet['removeURL'] = DataTweet['casefolding'].apply(preProcessingModul.removeURLemoji)  
    #==== Tokenisasi : memisahkan kata dalam kalimat
    DataTweet['Tokenisasi'] = DataTweet['removeURL'].apply(preProcessingModul.tokenize)    
    #====  Proses Casefolding2 hapus angka dan simbol
    DataTweet['Cleaning'] = DataTweet['Tokenisasi'].apply(preProcessingModul.hapus_simbolAngka)
    DataTweet2=pd.DataFrame()
    DataTweet2['Cleaning']=DataTweet['Cleaning']
    #============== Normalisasi: kata gaul, singkatan jadi kata baku    
    DataTweet['Normalisasi'] = DataTweet['Cleaning'].apply(preProcessingModul.normalisasi)
    #==== Stopword Removal : hapus kata yang tidak terlalu penting
    DataTweet['Stopword'] = DataTweet['Normalisasi'].apply(preProcessingModul.delstopwordID)
    #==== Stemming : mengurangi dimensi fitur kata/term
    DataTweet['Stemmed'] = DataTweet['Stopword'].apply(preProcessingModul.stemming)
    DataTweet['newTweet'] = DataTweet['Stemmed'].apply(preProcessingModul.listokalimat)

    #====================== lakukan TF-IDF
    savedtfidf = pickle.load(open("pickle_feature.pkl", 'rb'))
    vectorizer2 = TfidfVectorizer(vocabulary=savedtfidf)
    vect_docs2 = vectorizer2.fit_transform(DataTweet['newTweet'])
    features_names2 = vectorizer2.get_feature_names_out()
    
    dense2 = vect_docs2.todense()
    alist2 = dense2.tolist()
    newData2 = pd.DataFrame(alist2,columns=features_names2)
    # Load from file
    with open("pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    hasil=pickle_model.predict(newData2)
    DFpredict = pd.DataFrame(hasil,columns=["Prediksi"])
    prediksi=DFpredict.iloc[0]["Prediksi"]
    return prediksi

query = st.text_area("Masukan kalimat untuk diprediksi . . .", "")
if st.button('Say hello'):
    if(query!=""):
        hasil = prediksi(query)
        st.write(f"Hasil Prediksi : {hasil}")
    else:
        st.write("Masukan Kalimat Terlebih Dahulu")
    
