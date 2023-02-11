# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:26:49 2021

@author: AhmadYazidMunif
"""
import pandas as pd
import preProcessingModul
from sklearn.feature_extraction.text import TfidfVectorizer
from timeit import default_timer as timer
import pickle

start = timer()
#============== Read Data Input
stringe="aku iyaaa, katanya ga bisa memulai ulang kabar beranda padahal jaringan lumayann. ada juga yang gabisa kirim dm"
DataTweet = pd.DataFrame(data=[stringe], columns=['Tweet'])
#============== Start Processing Text
print("\n##-------- Mulai Proses Preprocessing --------##\n")
print('\n...... Proses Casefolding lowercase, hapus URL...... ')
DataTweet['casefolding'] = DataTweet['Tweet'].apply(preProcessingModul.lower)
DataTweet['removeURL'] = DataTweet['casefolding'].apply(preProcessingModul.removeURLemoji)
print(DataTweet)

#==== Tokenisasi : memisahkan kata dalam kalimat
print('\n...... Tokenisasi ...... ')
DataTweet['Tokenisasi'] = DataTweet['removeURL'].apply(preProcessingModul.tokenize)
print(DataTweet[['Tokenisasi']].head(2))

print('\n...... Proses Casefolding2 hapus angka dan simbol...... ')
DataTweet['Cleaning'] = DataTweet['Tokenisasi'].apply(preProcessingModul.hapus_simbolAngka)
DataTweet2=pd.DataFrame()
DataTweet2['Cleaning']=DataTweet['Cleaning']
print(DataTweet2[['Cleaning']].head(2))
#============== Normalisasi: kata gaul, singkatan jadi kata baku
print('\n...... Proses Normalisasi ...... ')
DataTweet['Normalisasi'] = DataTweet['Cleaning'].apply(preProcessingModul.normalisasi)
print(DataTweet[['Normalisasi']].head(2))

#==== Stopword Removal : hapus kata yang tidak terlalu penting
print('\n...... Proses Stopword Removal ...... ')
DataTweet['Stopword'] = DataTweet['Normalisasi'].apply(preProcessingModul.delstopwordID)
print(DataTweet[['Stopword']].head(6))
end1 = timer()
waktu1=end1 - start
print("Waktu Eksekusi = ",waktu1,"detik atau",waktu1/60,"menit")

#==== Stemming : mengurangi dimensi fitur kata/term
print('\n................ Proses Stemming ................ ')
start2 = timer()
DataTweet['Stemmed'] = DataTweet['Stopword'].apply(preProcessingModul.stemming)
print(DataTweet['Stemmed'].head(3))
DataTweet['newTweet'] = DataTweet['Stemmed'].apply(preProcessingModul.listokalimat)
end2 = timer()
waktu2=end2 - start2
print("Waktu Eksekusi Stemming = ",waktu2,"detik atau",waktu2/60,"menit")
#====================== lakukan TF-IDF
print('\n................ Hitung TF-IDF ................ ')
savedtfidf = pickle.load(open("pickle_feature.pkl", 'rb'))
vectorizer2 = TfidfVectorizer(vocabulary=savedtfidf)
vect_docs2 = vectorizer2.fit_transform(DataTweet['newTweet'])
features_names2 = vectorizer2.get_feature_names_out()
print(vect_docs2)

dense2 = vect_docs2.todense()
alist2 = dense2.tolist()
print('\n================')
newData2 = pd.DataFrame(alist2,columns=features_names2)
print(newData2)
# Load from file
with open("pickle_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)
hasil=pickle_model.predict(newData2)
DFpredict = pd.DataFrame(hasil,columns=["Prediksi"])
print(DFpredict.iloc[0]["Prediksi"])
gabungkan = pd.concat([DataTweet['Stemmed'], DFpredict], axis=1)
print(gabungkan)