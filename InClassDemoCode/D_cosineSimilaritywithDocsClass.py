# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:16:12 2018

@author: cvuppalapati
"""
# https://spacy.io/
# pip install spacy
# python -m spacy download en_core_web_sm
import string
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('stopwords')

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun")




tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix)
print(tfidf_matrix.shape)

print(tfidf_matrix[0:1])
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfidf_matrix[0:2], tfidf_matrix))
