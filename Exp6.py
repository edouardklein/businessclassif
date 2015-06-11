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
import pandas as pd

with open('labeled_firms_single.txt', 'r') as f:
    lines = f.read().split('\n')[1:]

cik2year_category = defaultdict(list)
category2year_cik = defaultdict(list)
all_categories = list(pd.read_excel('descriptive_words.xls')['SIC3'])
for l in [l for l in lines if l]:
    _, str_cik ,_, str_year, str_category = l.split('\t')
    category = int(str_category[2:-2])
    assert(category in all_categories)
    if category % 10 != 0:
        super_category = int(category/10)*10
    else:
        super_category = None # Category ends with '0', it already is a supercategory
    year = int(str_year)
    cik = int(str_cik)
    cik2year_category[cik].append([year, category])
    category2year_cik[category].append([year, cik])
    if super_category:
        cik2year_category[cik].append([year, super_category])
        category2year_cik[super_category].append([year, cik])

categories = list(category2year_cik.keys())
def nb_items(cat):
    return len(category2year_cik[cat])
categories.sort(key=nb_items)

print([[c,nb_items(c)] for c in categories])

with open('Exp2/classif_results.csv', 'w') as f:
    f.write('C, Accuracy, std_dev, TN, FP, FN, TP\n')
automatable_categories = []
for C in [c for c in categories if nb_items(c) >= 20]:
    print("\n\n****************\nCategory : "+str(C))
    C_text = []  # List of text samples for category C
    for year, cik in category2year_cik[C]:
        files = glob.glob('{}/{}-*'.format(year,cik))
        if not files:
            print('ACHTUNG: file for cik:{}, year:{} not found !'.format(cik, year))
            continue
        if len(files) > 1:
            print('ACHTUNG: more than one file for cik:{}, year:{} !'.format(cik, year))
            continue
        with open(files[0], 'r') as f:
            C_text.append(f.read())

            #print(C_text[12])
    N = len(C_text)
    print(str(N)+' samples acutally available !')


    categories_other_than_C = [c for c in category2year_cik.keys() if c!= C]
    other_year_cik = []
    other_text = []  # List of text samples for the other categories
    i=0
    while len(other_text) < len(C_text) and i<10000:
        i+=1
        not_C = random.choice(categories_other_than_C)
        year, cik = random.choice(category2year_cik[not_C])
        if [cik, year] in category2year_cik[C]: # We check it's not also in C
            print('cik:{}, year:{} also in category {}, choosing another one'.format(cik, year, C))
            continue
        files = glob.glob('{}/{}-*'.format(year,cik))
        if not files:
            print('ACHTUNG: file for cik:{}, year:{} not found !'.format(cik, year))
            continue
        if len(files) > 1:
            print('ACHTUNG: more than one file for cik:{}, year:{} !'.format(cik, year))
            continue
        other_year_cik.append([year, cik])
        with open(files[0], 'r') as f:
            other_text.append(f.read())
        print('Adding cik:{}, year:{}, total length now {}'.format(cik, year, len(other_text)))
    print(str(len(other_text))+' samples not in category '+str(C))



    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    all_text = C_text + other_text
    Y = np.zeros(len(C_text) + len(other_text))
    Y[:len(C_text)] = 1
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(all_text)
    print(X_counts.shape)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print(X_tfidf.shape)


    def clf_eval(clf):
        loo = cross_validation.LeaveOneOut(len(Y))
        #Accuracy
        scores = cross_validation.cross_val_score(clf, X_tfidf, Y, cv=10)
        acc = scores.mean()
        std = scores.std()
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        #Confusion Matrix
        Y_true = []
        Y_pred = []
        misclassified = []
        for train_index, test_index in loo:
            X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, Y_train)
            Y_pred.append(clf.predict(X_test))
            Y_true.append(Y_test)
            if Y_true[-1] != Y_pred[-1]:
                assert(len(test_index) == 1)
                misclassified_index = test_index[0]
                misclassified_index -= len(C_text)
                misclassified.append(other_year_cik[misclassified_index])
        cm =  confusion_matrix(Y_true, Y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig('Exp2/'+str(C)+'confusion_matrix.pdf')
        plt.close()
        return acc, std, cm, misclassified


    clf = MultinomialNB()
    acc, std, cm, misclassified = clf_eval(clf)
    if acc - std >= .7:  # Our cutoff for what constitute 'good performance'
        with open('Exp2/'+str(C)+'_misclassified.txt', 'w') as f:
            for year, cik in misclassified:
                f.write('YEAR {}, CIK {}, ______________________________\n\n'.format(year, cik))
                files = glob.glob('{}/{}-*'.format(year,cik))
                with open(files[0], 'r') as f2:
                    f.write(f2.read())
        with open('Exp2/classif_results.csv', 'a') as f:
            #Headers are written in the file just before the for loop
            f.write(','.join(map(str,[C, acc, std]+list(cm.reshape(-1))))+'\n')
        with open('Exp2/'+str(C)+'_count_vect.pickle', 'wb') as f:
            pickle.dump(count_vect, f)
        with open('Exp2/'+str(C)+'_tfidf_transformer.pickle', 'wb') as f:
            pickle.dump(tfidf_transformer, f)
        clf.fit(X_tfidf, Y)
        with open('Exp2/'+str(C)+'_classifier.pickle', 'wb') as f:
            pickle.dump(clf, f)
        automatable_categories.append(C)

nb_super = len([x for x in all_categories if x%10 == 0])
automatable_super = len([x for x in automatable_categories if x%10 == 0])
print("We can 'reliably' recognize {} out of {} supercategories.".format(automatable_super, nb_super))

nb_sub = len([x for x in all_categories if x%10 != 0])
automatable_sub = len([x for x in automatable_categories if x%10 != 0])
print("We can 'reliably' recognize {} out of {} categories.".format(automatable_sub, nb_sub))
