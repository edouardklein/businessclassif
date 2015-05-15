from collections import defaultdict
import glob
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

count_vectorizers = {}
for f in glob.glob('Exp2/*_count_vect.pickle'):
    cat = int(f.split('_')[0])
    count_vectorizers[cat] = pickle.load(open(f, 'rb'))

tfidf_transformers = {}
for f in glob.glob('Exp2/*_tfidf_transformer.pickle'):
    cat = int(f.split('_')[0])
    tfidf_transformers[cat] = pickle.load(open(f, 'rb'))

classifiers = {}
for f in glob.glob('Exp2/*_classifier.pickle'):
    cat = int(f.split('_')[0])
    classifiers[cat] = pickle.load(open(f, 'rb'))

assert(classifiers.keys() == tfidf_transformers.keys())
categories = classifiers.keys()

years = ['2001']
companies_files = sum([glob.glob(year+'/*.txt') for year in years], [])

#print(companies_files[:100])

with open('Exp2/BUS_labels_from_classifiers.txt', 'w') as f:
    f.write('cik\tyear\tcat\n')
for file in companies_files:
    with open(file, 'r') as f:
        text = f.read()
    cik = file.split('\\')[1].split('-')[0]
    year = file.split('\\')[0]
    for cat in categories:
        count = count_vectorizers[cat]
        tfidf = tfidf_transformers[cat]
        clf = classifiers[cat]
        X = tfidf.transform(count.transform(text))
        if clf.predict(tfidf.transform(count.transform([text]))) == 1:
            with open('BUS_labels_from_classifiers.txt', 'a') as f:
                f.write('\t'.join([cik, year, str(cat)])+'\n')


