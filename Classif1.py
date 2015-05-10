# These are the modules I use, If you use Anaconda, they should be installed on your system.
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

# Here I open the 'labeled_firms_single.txt' file and creat a list called 'lines' from it, each element is a line of the file
with open('labeled_firms_single.txt', 'r') as f:
    lines = f.read().split('\n')[1:]
#From this list I create two maps : one maps one cik to the list of categories it beloings to
#the other maps the categories to the list of cik belonging to that category
#We also put the year (1998 or 1999) in the mapped elements, in order to know in which folder to look for the file.
cik2year_category = defaultdict(list)
category2year_cik = defaultdict(list)
for l in [l for l in lines if l]:
    _,cik,_,year,category = l.split('\t')
    category = int(category[2:-2])
    year = int(year)
    cik = int(cik)
    cik2year_category[cik].append([year, category])
    category2year_cik[category].append([year, cik])

#Here we sort the categories by the number of companies that belong to it, in order to study the categories with the most samples
categories = list(category2year_cik.keys())
def nb_items(cat):
    return len(category2year_cik[cat])
categories.sort(key=nb_items)


#This will display the number of examples we have for each category
print([[c,nb_items(c)] for c in categories])

#Now, we create the C_text list. Each element is the content of one text file (in either folder 1999/ or 1998/)
#associated with category C (you can change the value of C to study another category)
C = 602#284#602#284#602
C_text = []  # List of text samples for category C
for year, cik in category2year_cik[C]:
    # The following line create a list called 'files' which contains all files whose name starts with '<cik>-' in folder <year>
    # of course, <cik> and <year> mean the values of those variables, e.g. 1998 for <year>
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
print(N)


#Now that C_text contains the text of the companies of category C, we need samples for other categories that are not C.
#We create the list 'other_text' to hold those textual samples
categories_other_than_C = [c for c in category2year_cik.keys() if c!= C]
other_year_cik = []
other_text = []  # List of text samples for the other categories
i=0
while len(other_text) < len(C_text) and i<10000:  #While we don't have as much samples for 'not C' as we have for C
    i+=1
    not_C = random.choice(categories_other_than_C)  #We choose a category at random
    year, cik = random.choice(category2year_cik[not_C])  #We choose a cik (and the associated year) at random in this category
    if [cik, year] in category2year_cik[C]: # We check it's not also in C
        print('cik:{}, year:{} also in category {}, choosing another one'.format(cik, year, C))
        continue
    files = glob.glob('{}/{}-*'.format(year,cik))  #We open the corresponding file (same as above with <year>/<cik>-whatever)
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
print(len(other_text))



#This is almost straight from http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#We convert the text elements of lists C_text and other_text to vectors of numbers using the tf-idf transform.
all_text = C_text + other_text
Y = np.zeros(len(C_text) + len(other_text))  #The Y vector contains the 'answer' : its i-th element is 1 if
# Element i of X_* is in category C, 0 if not.
Y[:len(C_text)] = 1
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(all_text)
print(X_counts.shape)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print(X_tfidf.shape)


#This function prints the accuracy of the classifier, and displays the confusion matrix, see :
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
def clf_eval(clf):
    loo = cross_validation.LeaveOneOut(len(Y))
    #Accuracy
    scores = cross_validation.cross_val_score(clf, X_tfidf, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #Confusion Matrix
    Y_true = []
    Y_pred = []
    for train_index, test_index in loo:
        X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #w = weights[train_index]
        #clf.fit(X_train, Y_train, sample_weight=w)
        clf.fit(X_train, Y_train)
        Y_pred.append(clf.predict(X_test))
        Y_true.append(Y_test)
    cm =  confusion_matrix(Y_true, Y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.pdf')
    return cm


# We train the classifier and evaluate its performance using a k-fold
# http://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
clf = MultinomialNB()
clf_eval(clf)