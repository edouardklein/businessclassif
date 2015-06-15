import pandas as pd
from data_load import *
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

desc_table = pd.read_excel('descriptive_words.xls')
d_words = {}
for cat in all_categories:
    row = desc_table[desc_table['SIC3'] == cat]
    d_words[cat] = row[row.columns[2:]].dropna(axis=1).loc[row.index[0]].tolist()


results = read_pickle('results', alt={}, prefix='Exp7/')
def load_or_compute_results(method_str, method):
    '''Load or compute the results for the specified method'''
    if  method_str+'/ycik2scats' in results and method_str+'/scat2yciks' in results:
        return
    ycik2scats = {}
    scat2yciks = defaultdict(set)
    for i, (year, cik) in enumerate(all_yciks):
        print('INFO: Running strategy '+method_str+' on firm #{}'
              'of {}'.format(i, len(all_yciks)))
        scats = method(year, cik, cats=all_supercategories)
        ycik2scats[(year, cik)] = set(scats)
        for scat in scats:
            scat2yciks[scat].add((year, cik))
    results[method_str+'/ycik2scats'] = ycik2scats
    results[method_str+'/scat2yciks'] = scat2yciks
    write_pickle('results', results, prefix='Exp7/')

# cat2regexp[XX0] will yield the words specified only in the line corresponding to
# XX0 in the excel file
# whereas scat2regexp[XX0] will yield those words, as well as all the words for the
# subcategories XX1, XX2, etc.
cat2regexp = {cat: re.compile('('+'|'.join(d_words[cat])+')')
                for cat in all_categories}
scat2regexp = {}
for scat in all_supercategories:
    words = []
    for cat in [c for c in all_categories if supercat(c) == scat]:
        words += d_words[cat]  # This includes scat as well
    scat2regexp[scat] = re.compile('('+'|'.join(words)+')')


def any_match(year, cik, cats=all_categories):
    '''Return all categories within cats for which any descriptive word
    is in the text'''
    text = year_cik2text[(year, cik)]
    answer = []
    for cat in cats:
        if cat2regexp[cat].search(text):
            answer.append(cat)
    return answer


def any_aggregated_match(year, cik, scats=all_supercategories):
    '''Return all supercategories for which any of aggregated (i.e. with the
    subcategories' words as well) words is in the text'''
    text = year_cik2text[(year, cik)]
    answer = []
    for scat in scats:
        if scat2regexp[scat].search(text):
            answer.append(scat)
    return answer


load_or_compute_results('any_match', any_match)


def all_must_match(year, cik, cats=all_categories):
    '''Return all categories for which all descriptive words are in the text'''
    text = year_cik2text[(year, cik)]
    answer = []
    for cat in cats:
        add = True
        for word in d_words[cat]:
            if not word in text:
                add = False
                break
        if add:
            answer.append(cat)
    return answer


load_or_compute_results('all_must_match', all_must_match)


def n_most_matching(year, cik, n, cats=all_categories):
    '''Return the n categories with the highest percentage of matching words'''
    text = year_cik2text[(year, cik)]
    matching_prop = {}  # \in [0;1], the proportion of descriptive words in the text
    for cat in cats:
        matching_prop[cat] = 0
        for word in d_words[cat]:
            if word in text:
                matching_prop[cat] += 1
        matching_prop[cat] /= len(d_words[cat])
    sorted_cats = sorted(list(matching_prop.keys()), key=lambda k: matching_prop[k], reverse=True)
    return sorted_cats[0:n+1]


def most_matching(year, cik, cats=all_categories):
    return n_most_matching(year, cik, 1, cats=cats)


def most_2_matching(year, cik, cats=all_categories):
    return n_most_matching(year, cik, 2, cats=cats)


def most_5_matching(year, cik, cats=all_categories):
    return n_most_matching(year, cik, 5, cats=cats)


def most_20_matching(year, cik, cats=all_categories):
    return n_most_matching(year, cik, 20, cats=cats)


load_or_compute_results('Most_matching', most_matching)
load_or_compute_results('2_most_matching', most_2_matching)
load_or_compute_results('5_most_matching', most_5_matching)
load_or_compute_results('20_most_matching', most_20_matching)


def match_within_rank_k(year, cik, k, cats=all_categories):
    '''Return the categories for which any descriptive word has rank <= k in the company's description'''
    text = year_cik2text[(year, cik)]
    count_vect = CountVectorizer(stop_words='english')
    X = list(count_vect.fit_transform([text]).toarray().reshape(-1))
    voc = {v: k for k, v in count_vect.vocabulary_.items()}
    words_of_rank_k = []
    for _ in range(0,k):
        index = np.argmax(X)
        words_of_rank_k.append(voc[index])
        X.pop(index)
        if len(X) == 0:
            break  # Some texts have less than 30 words !
    words_of_rank_k = set(words_of_rank_k)
    answer = []
    for cat in cats:
        if len(words_of_rank_k & set(d_words[cat])) > 0:
            answer.append(cat)
    return answer


def match_within_rank_1(year, cik, cats=all_categories):
    return match_within_rank_k(year, cik, 1, cats=cats)


def match_within_rank_10(year, cik, cats=all_categories):
    return match_within_rank_k(year, cik, 10, cats=cats)


def match_within_rank_30(year, cik, cats=all_categories):
    return match_within_rank_k(year, cik, 30, cats=cats)


def match_within_rank_100(year, cik, cats=all_categories):
    return match_within_rank_k(year, cik, 100, cats=cats)


load_or_compute_results('match_within_rank_1', match_within_rank_1)
load_or_compute_results('match_within_rank_10', match_within_rank_10)
load_or_compute_results('match_within_rank_30', match_within_rank_30)
load_or_compute_results('match_within_rank_100', match_within_rank_100)
