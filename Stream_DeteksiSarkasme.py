# Import library yang diperlukan
import streamlit as st
import pandas as pd
import pickle
import string
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Download resources dari NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi pra-pemrosesan teks
def lower(text):
    # lowercase
    lower = text.lower()
    return lower

def removeURLemoji(text):
    # hapus hastag/mention
    HastagRT = re.sub(r"#(\w+)|@(\w+)|(\brt\b)", " ", text)
    # hapus URL
    pola_url = r'http\S+'
    CleanURL = re.sub(pola_url, " ", HastagRT)
    # hapus emoticon
    hps_emoji = hapus_emoticon(CleanURL)
    # hapus multiWhitespace++, ex: ahh   haa
    text = re.sub('\s+', ' ', hps_emoji)
    # hasil akhir casefolding
    hasil = text
    return hasil

def angkadua(teksAwal2):
    final2 = []
    huruf2 = ""
    for x in range(len(teksAwal2)):
        cek2 = [i for i in teksAwal2[x]]
        for x in range(len(cek2)):
            if x == 0:
                final2.append(cek2[0])
                huruf2 = cek2[0]
            else:
                if cek2[x] != huruf2:
                    if cek2[x] == "2":
                        if(len(final2)) == 2:
                            final2.append(cek2[x-2])
                            final2.append(cek2[x-1])
                            huruf2 = cek2[x]
                        elif(len(final2) > 2):
                            jo = "".join(cek2[:2])
                            if(jo == "se" or jo == "di"):
                                final2.append(" ")
                                final2 = final2+cek2[2:x]
                                huruf2 = cek2[x]
                            else:
                                final2.append(" ")
                                final2 = final2+cek2[:x]
                                huruf2 = cek2[x]
                        else:
                            final2.append(cek2[x])
                            huruf2 = cek2[x]
                    else:
                        final2.append(cek2[x])
                        huruf2 = cek2[x]
                else:
                    final2.append(cek2[x])
                    huruf2 = cek2[x]
        final2.append(" ")
    hasil = "".join(final2).split()
    return hasil


def hapus_hurufganda(teksAwal):
    jml = 0

    final = []
    huruf = ""
    for x in range(len(teksAwal)):
        cek = [i for i in teksAwal[x]]
        for x in range(len(cek)):
            if x == 0:
                final.append(cek[0])
                huruf = cek[0]
                jml = 1
            else:
                if cek[x] != huruf:
                    final.append(cek[x])
                    huruf = cek[x]
                    jml = 1
                else:
                    if jml < 2:
                        final.append(cek[x])
                        huruf = cek[x]
                        jml += 1
        final.append(" ")
    hasil = "".join(final).split()
    return hasil


def hapus_simbolAngka(text):
    del_angkadua = angkadua(text)
    del_hrfganda = hapus_hurufganda(del_angkadua)

    # hasil=[]
    token = del_hrfganda
    lte = ["2g", "3g", "4g", "5g"]
    for i in range(len(token)):
        if(token[i] not in lte):
            token[i] = re.sub(r"\d+", " ", token[i])

    for ele in range(len(token)):
        token[ele] = token[ele].translate(
            str.maketrans('', '', string.punctuation))
        token[ele] = re.sub('\W', "", token[ele])
        token[ele] = re.sub('\s+', "", token[ele])

    return token


def hapus_simbolAngka2(text):
    token = text
    for i in range(len(token)):
        cekG = re.match(r"([\b234]+g)", token[i])
        if (cekG) == None:
            token[i] = re.sub(r"\d+", "", token[i])
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in token:
        if ele in punc:
            token = token.replace(ele, " ")
            token = re.sub('\s+', ' ', token)
    return token


def hapus_emoticon(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    # hapus emoji
    CleanEmoji = re.sub(emoji_pattern, "", text)
    return CleanEmoji


def tokenize(kalimat):
    return word_tokenize(kalimat)


def listokalimat(kalimat):
    listToStr = ' '.join(kalimat)
    return listToStr


def delstopwordID(teks):
    notsinglechar=[]
    for kata in teks:
        a = re.sub(r"\b[a-zA-Z]\b", " ", kata)
        if(a!=" "):
            notsinglechar.append(a)
    return [kata for kata in notsinglechar if kata not in list_stopwords]


def daftarStopword():
    list_stopwords = stopwords.words('indonesian')
    #list_stopwords=nltk.corpus.stopwords.words("indonesian")
    # baca tambahan
    tambahan = ['&amp', 'an', 'anms', 'anu', 'by', 'cc', 'dll', 'do', 'dst', 'fi', 'hoi', 'ic', 'id', 'in', 'jo', 'kah', 'kan', 'ke', 'klab', 'lik', 'lnk', 'meng', 'mer', 'mer', 'mu', 'nt', 'nya', 'opt', 'per', 'pu', 're', 're', 'rp', 'rpan', 'rza', 'se', 'ter', 'the', 'tj', 'tk', 'tl', 'un', 'wi', 'xx']
    list_stopwords.extend(tambahan)
    list_stopwords = set(list_stopwords)
    return list_stopwords


def normal_term():
    normalisasi_word = pd.read_excel("_normalisasi.xlsx")
    normalisasi_dict = {}
    for index, row in normalisasi_word.iterrows():
        if row[0] not in normalisasi_dict:
            normalisasi_dict[row[0]] = row[1]
    return normalisasi_dict


def normalisasi(document):
    kalimat = document
    for term in range(len(kalimat)):
        if kalimat[term] in normalisasi_dict:
            kalimat[term] = normalisasi_dict[kalimat[term]]
    hasil = " ".join(kalimat).split()
    return hasil


def stemming(kalimat):
    term_dict = {}
    for kata in kalimat:
        for term in kalimat:
            if term not in term_dict:
                term_dict[term] = " "
    temp = list(term_dict)
    for x in range(len(temp)):
        if temp[x] == "jaringan":
            term_dict[temp[x]] = temp[x]
        else:
            term_dict[temp[x]] = stemmer.stem(temp[x])
    kalimat = [term_dict[term] for term in kalimat]
    #listToStr = ' '.join([str(i) for i in kalimat])
    return kalimat

list_stopwords = daftarStopword()
term_dict = {}
factory = StemmerFactory()
stemmer = factory.create_stemmer()
normalisasi_dict = normal_term()

# Fungsi prediksi
def prediksi(q):
    DataTweet = pd.DataFrame(data=[q], columns=['Tweet'])
    # Proses pra-pemrosesan teks
    DataTweet['casefolding'] = DataTweet['Tweet'].apply(lower)
    DataTweet['removeURL'] = DataTweet['casefolding'].apply(removeURLemoji)  
    DataTweet['Tokenisasi'] = DataTweet['removeURL'].apply(tokenize)    
    DataTweet['Cleaning'] = DataTweet['Tokenisasi'].apply(hapus_simbolAngka)
    DataTweet2 = pd.DataFrame()
    DataTweet2['Cleaning'] = DataTweet['Cleaning']
    DataTweet['Normalisasi'] = DataTweet['Cleaning'].apply(normalisasi)
    DataTweet['Stopword'] = DataTweet['Normalisasi'].apply(delstopwordID)
    DataTweet['Stemmed'] = DataTweet['Stopword'].apply(stemming)
    DataTweet['newTweet'] = DataTweet['Stemmed'].apply(listokalimat)

    # Lakukan TF-IDF
    savedtfidf = pickle.load(open("pickle_feature.pkl", 'rb'))
    vectorizer2 = TfidfVectorizer(vocabulary=savedtfidf)
    vect_docs2 = vectorizer2.fit_transform(DataTweet['newTweet'])
    features_names2 = vectorizer2.get_feature_names_out()
    
    dense2 = vect_docs2.todense()
    alist2 = dense2.tolist()
    newData2 = pd.DataFrame(alist2, columns=features_names2)
    
    # Load model dan lakukan prediksi
    with open("pickle_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    hasil = pickle_model.predict(newData2)
    predicted = hasil[0]
    return predicted
def plot_distribution_chart(data):
    st.subheader('Visualisasi Sebaran Data')

    if 'Label' in data.columns:  # Ubah 'Label' dengan nama kolom label Anda
        value_counts = data['Label'].value_counts()
        st.bar_chart(value_counts)
    else:
        st.write("Kolom 'Label' tidak ditemukan dalam dataset.")

def side():
    with st.sidebar:
        genre = st.radio(
            "Pilih Menu",
            ('Implementasi', 'Dataset', 'Grafik')
        )
        st.write(f"Anda memilih {genre}.")
        return genre

def implementasi():
    st.subheader('#Testing-ProjectModel')
    st.write("""
    # Deteksi Sarkasme
    Uji Coba Hanya Mendukung Bahasa Indonesia 
    """)
    query = st.text_area("Masukan kalimat untuk diprediksi . . .", "")
    if st.button('Prediksi', key="prediksi"):
        if query:
            hasil = prediksi(query)
            st.write(f"Hasil Prediksi : {hasil}")
        else:
            st.write("Masukan Kalimat Terlebih Dahulu")

def dataset():
    st.subheader('#Dataset')
    st.write('Ini merupakan potongan dari dataset. Hasil pelatihan (training) menggunakan Support Vector Machine Ensemble mencapai nilai akurasi 77.3%')
    namaFile = "#tweet.xlsx"
    DataTweet2 = pd.read_excel('#tweet.xlsx')
    st.dataframe(DataTweet2)

def grafik():
    st.subheader('Grafik Sebaran Data')
    namaFile = "#tweet.xlsx"
    DataTweet2 = pd.read_excel('#tweet.xlsx')
    plot_distribution_chart(DataTweet2)

def run_app():
    choice = side()

    if choice == 'Implementasi':
        implementasi()
    elif choice == 'Dataset':
        dataset()
    else:
        grafik()

if __name__ == "__main__":
    run_app()