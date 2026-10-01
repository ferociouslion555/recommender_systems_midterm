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

nltk.download('stopwords')

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')
    
# Determine semantic similarities
print("\n Determine semantic similarities\n")
doc1 = nlp(u"my fries were super gross")

print("Doc1\n")
print(doc1)
doc2 = nlp(u"such disgusting fries")
print("Doc2\n")
print(doc2)

similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)