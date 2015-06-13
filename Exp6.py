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
import functools


# Loading labelled data
with open('labeled_firms_single.txt', 'r') as f:
    lines = f.read().split('\n')[1:]
all_categories = list(pd.read_excel('descriptive_words.xls')['SIC3'])  # All categories that exist


def is_super(cat):
    '''Return True if cat is a supercategory'''
    return cat % 10 == 0


def is_sub_in_sup(sub, sup):
    '''Return True if (super)category sub is in supercategory sup'''
    assert is_super(sup), 'sup is not a supercategory'
    return int(sub/10)*10 == sup
super_categories = [c for c in all_categories if is_super(c)]


def cat_equal(c1,c2):
    '''Return true if c1 == c2 or c1 in c2 or c2 in c1'''
    if is_super(c1):
        return is_sub_in_sup(c2, c1)
    elif is_super(c2):
        return is_sub_in_sup(c1, c2)
    else:
        return c1 == c2


def should_delete(category):
    '''Whether a company should be deleted from the labelled data'''
    return is_super(category) or category == 999

#DEBUG REMOVE
DEBUG_categories = [602, 283, 737, 603, 679, 384]
DEBUG_firms = defaultdict(set)
#END DEBUG
category2year_cik = defaultdict(set)
year_cik2category = defaultdict(set)
for l in [l for l in lines if l]:
    _, str_cik ,_, str_year, str_category = l.split('\t')
    category = int(str_category[2:-2])
    assert(category in all_categories)
    year = int(str_year)
    cik = int(str_cik)
    year_cik2category[(year, cik)].add(category)
    #DEBUG REMOVE
    if category in DEBUG_categories:
        DEBUG_firms[(year, cik)].add(category)
    #END DEBUG

old_length = len(year_cik2category)
year_cik2category = {k:year_cik2category[k] for k in year_cik2category if not any(
    [should_delete(cat) for cat in year_cik2category[k]])}
print('INFO: We removed {} firms because one of their labels '
      'was a supercategory or 999'.format(old_length - len(year_cik2category)))

for ycik,cat_set in year_cik2category.items():
    for cat in cat_set:
        category2year_cik[cat].add(ycik)
categories = list(category2year_cik.keys())  # Categories for which we hope to have data


@functools.lru_cache()
def ycik2text(year, cik, not_cat=None, disjoint_from=None):
    '''Return the text of (cik, year) under the given constraints, and None
    if no such text exists'''
    if not_cat:
        if is_super(not_cat):
            forbidden = [c for c in categories if is_sub_in_sup(c, not_cat)]
        else:
            forbidden = [not_cat]
        for c in forbidden:
            if (cik, year) in category2year_cik[c]:
                print('DEBUG: cik:{}, year:{} also in category {},'
                      ' choosing another one'.format(cik, year, not_cat))
                return None
    if disjoint_from and (cik, year) in disjoint_from:
        print('DEBUG:  cik:{}, year:{} in the forbidden set,'
              ' choosing another one'.format(cik, year))
        return None
    files = glob.glob('{}/{}-*'.format(year,cik))
    if not files:
        print('WARNING: file for cik:{}, year:{} not found !'.format(cik, year))
        return None
    if len(files) > 1:
        print('WARNING: more than one file for '
              'cik:{}, year:{} !'.format(cik, year))
        return None
    return open(files[0], 'r').read()


@functools.lru_cache()
def text_samples_for_category(cat):
    """Return, as a list, the text samples for (super)category c"""
    #FIXME: Make it work with supercategory
    C_text = []
    C_year_cik = []
    cats = [c for c in categories if cat_equal(cat, c)]  # WARNING : c cannot be a
    # supercategory because of should_delete(), if that changes, this
    # line may be wrong
    for c in cats:
        for year, cik in category2year_cik[c]:
            text = ycik2text(year, cik)
            if not text:
                continue
            C_text.append(text)
            C_year_cik.append((year, cik))
    return C_text, C_year_cik


def nb_items(cat):
    return len(text_samples_for_category(cat)[0])

#DEBUG REMOVE
categories = [602, 283, 737, 603, 679, 384]
#END DEBUG
for i,c in enumerate(categories):
    #Populating the cache, while printing progress
    print('INFO: Loading text samples for category {} of {}'.format(i, len(categories)))
    text_samples_for_category(c)

categories = [c for c in categories if nb_items(c) > 1]  # With only 1 sample,
# no leave-one-out is possible, categories is now the list of categories we can train on
categories.sort(key=nb_items)
print([[c,nb_items(c)] for c in categories])



@functools.lru_cache()
def text_samples_for_category_other_than(cat, n, disjoint_from=frozenset([])):
    """Return a list of n text samples for firms not in (super)category c"""
    categories_other_than_cat = [c for c in categories if not cat_equal(c, cat)]
    other_year_cik = []
    other_text = []  # List of text samples for the other categories
    i=0
    while len(other_text) < n and i<10000:
        i+=1
        not_cat = random.choice(categories_other_than_cat)
        year, cik = random.choice(list(category2year_cik[not_cat]))
        text = ycik2text(year, cik, not_cat=cat, disjoint_from=disjoint_from)
        if not text or (year, cik) in other_year_cik:
            continue
        other_year_cik.append((year, cik))
        other_text.append(text)
        #print('DEBUG: Adding cik:{}, year:{},'
        #      ' total length now {}'.format(cik, year, len(other_text)))
    print('INFO: '+str(len(other_text))+' samples not in category '+str(cat))
    assert len(other_text) == n, "We somehow could not find "
    "{} negative samples".format(n)
    return other_text, other_year_cik


def clf_single_eval(clf, ycik, year_cik2clf_yes, cat2year_cik):
    '''Update the two dicts with the results of asking clf to classify ycik'''
    text = ycik2text(ycik[0], ycik[1])
    if not text:  # No sample text, returning the dicts unchanged
        return year_cik2clf_yes, cat2year_cik
    a = clf.predict(text)
    if a == 0:
        return year_cik2clf_yes, cat2year_cik
    # a == 1
    if (year, cik) in year_cik2clf_yes:
        year_cik2clf_yes[(year, cik)].append(clf.category)
    else:
        year_cik2clf_yes[(year, cik)] = [clf.category]
    if clf.category in cat2year_cik:
        cat2year_cik[clf.category].append((year, cik))
    else:
        cat2year_cik[clf.category] = [(year, cik)]
    return year_cik2clf_yes, cat2year_cik


def read_pickle(fname, alt=None):
    '''Read a pickled variable from a file, returns the non None provided alt if
    not found, raise an exception if not found and alt is None'''
    try:
         with open('Exp6/'+fname+'.pickle', 'rb') as f:
             return pickle.load(f)
    except FileNotFoundError:
        if not alt is None:
            return alt
        raise


def write_pickle(fname, var):
    '''Serialize var into the file named fname'''
    with open('Exp6/'+fname+'.pickle', 'wb') as f:
        pickle.dump(var, f)


class TextClassifer:
    def __init__(self, cat, cv, tf, clf):
        self.category = cat
        self.count_vect = cv
        self.tfidf = tf
        self.clf = clf

    def predict(self, txt):
        X_counts = self.count_vect.transform([txt])
        X_tfidf = self.tfidf.transform(X_counts)
        return self.clf.predict(X_tfidf)


def train_clf(cat, pos_samples, neg_samples, save=None):
    '''Train a classifier on the given text samples'''
    try:
        return read_pickle(str(cat)+'_'+save+'_clfbunch')
    except( FileNotFoundError, TypeError):
        pass
    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    all_text = pos_samples + neg_samples
    Y = np.zeros(len(pos_samples) + len(neg_samples))
    Y[:len(pos_samples)] = 1
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(all_text)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    clf = MultinomialNB()
    clf.fit(X_tfidf, Y)
    answer = TextClassifer(cat, count_vect, tfidf_transformer, clf)
    if save:
        write_pickle(str(cat)+'_'+save+'_clfbunch', answer)
    return answer


def clf_eval(c):
    """Compute useful metrics for a classifier trained on (super)category c"""
    # The return values
    year_cik2clf_yes = read_pickle('year_cik2clf_yes', alt={})  # maps
    # cik to a list of classifiers that recognized its text as belonging to their
    # (super)category
    cat2year_cik = read_pickle('cat2year_cik', alt={})  # maps a
    # (super)category to the list of (year, cik) that matched.
    if c in cat2year_cik:
        return year_cik2clf_yes, cat2year_cik
    # Create base training set
    C_texts, C_ycik = text_samples_for_category(c)
    N = len(C_texts)
    notC_texts, notC_ycik = text_samples_for_category_other_than(c, N)
    # Create alternative training set
    alt_notC_texts, alt_notC_ycik = text_samples_for_category_other_than(c,
                                                                         N,
                                                                         disjoint_from=
                                                                         tuple(notC_ycik))
    # Train on both
    main_clf = train_clf(c, C_texts, notC_texts, save='main')
    alt_clf = train_clf(c, C_texts, alt_notC_texts, save='alt')
    # For every firm
    #for i, (ycik, cat) in enumerate(year_cik2category.items()):
    for i, (ycik, cat) in enumerate(DEBUG_firms.items()):
        print('DEBUG: Evaluation on firm # {} of {}'.format(i, len(year_cik2category)))
        # If not in training set
        if not ycik in C_ycik and not ycik in notC_ycik:
            # classify on main classfier
            clf = main_clf
        # If in the training set as a negative example
        elif ycik in notC_ycik:
            # classify on alternative classifier
            clf = alt_clf
        # If in the training set as a positive example
        elif ycik in C_ycik:
            # Create new training set without the sample
            i = C_ycik.index(ycik)
            if i < len(C_texts) - 1:
                alt_C_texts = C_texts[0:i] + C_texts[i+1:]
            else:
                alt_C_texts = C_texts[0:i]
            # Train and classify
            clf = train_clf(c, alt_C_texts, notC_texts)
        year_cik2clf_yes, cat2year_cik = clf_single_eval(clf, ycik,
                                                         year_cik2clf_yes,
                                                         cat2year_cik)
    write_pickle('year_cik2clf_yes', year_cik2clf_yes)
    write_pickle('cat2year_cik', cat2year_cik)
    return year_cik2clf_yes, cat2year_cik

#FIXME: if main blah
#FIXME code to compare to ground truth and print the results as well and pickle them in clf_eval
scs = set([int(c/10)*10 for c in categories])
for i, sc in enumerate(scs):
    print('INFO: Eval for category {}, {}/{}'.format(sc, i, len(scs)))
    clf_eval(sc)
