# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:48:05 2021

@author: SelimPc
"""




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
from snowballstemmer import TurkishStemmer

#Import file
all_txt_files =[]
for file in Path("./input/dataset/").rglob("*.txt"):
     all_txt_files.append(file.parent / file.name)

all_docs = []
turkStem=TurkishStemmer()
#Import Data and Data Stemming
for txt_file in all_txt_files:
    with open(txt_file,encoding="utf8") as f:
        txt= []
        txt_file_as_string = f.read()
        tokenizer = RegexpTokenizer('\w+')
        words = tokenizer.tokenize(txt_file_as_string) 
        for word in words:
           txt.append(turkStem.stemWord(str(word)))
    all_docs.append(str(txt))
    
#idf weights values
doc = CountVectorizer(lowercase=True)
word_count=doc.fit_transform(all_docs)
print("Word Count")
print(word_count)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=doc.get_feature_names(),columns=["idf_weights"])
print("IDF_WEIGHTS")
print(df_idf.sort_values(by=['idf_weights']).head(10))

#tfidf
tfidf_vectorizer=TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True,lowercase=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(all_docs)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print("TFIDF FOR bilgisayar.txt")
print(df.sort_values(by=["tfidf"],ascending=False).head(10))
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[1]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print("TFIDF FOR film.txt")
print(df.sort_values(by=["tfidf"],ascending=False).head(10))
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[2]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print("TFIDF FOR kozmetik_kisisel_bakim.txt")
print(df.sort_values(by=["tfidf"],ascending=False).head(10))
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[3]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print("TFIDF FOR otoaksesuar.txt")
print(df.sort_values(by=["tfidf"],ascending=False).head(10))
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[4]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print("TFIDF FOR telefon.txt")
print(df.sort_values(by=["tfidf"],ascending=False).head(10))







