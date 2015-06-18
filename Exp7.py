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


results = read_pickle('results', alt={'methods':[]}, prefix='Exp7/')
results['colors'] = []
def load_or_compute_results(method):
    '''Load or compute the results for the specified method'''
    if  method in results['methods']:
        return
    yk2scats = {}
    scat2yks = defaultdict(set)
    method_func = eval(method)
    for i, (year, cik, gvkey) in enumerate(all_yks):
        print('INFO: Running strategy '+method+' on firm #{}'
              'of {}'.format(i, len(all_yks)))
        scats = method_func(year, cik, gvkey, cats=known_supercategories)
        yk2scats[(year, cik, gvkey)] = set(scats)
        for scat in scats:
            scat2yks[scat].add((year, cik, gvkey))
    results[method+'/yk2scats'] = yk2scats
    results[method+'/scat2yks'] = scat2yks
    results['methods'].append(method)
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


def any_match(year, cik, gvkey, cats=all_categories):
    '''Return all categories within cats for which any descriptive word
    is in the text'''
    text = year_key2text[(year, cik, gvkey)]
    answer = []
    for cat in cats:
        if cat2regexp[cat].search(text):
            answer.append(cat)
    return answer


def any_aggregated_match(year, cik, gvkey, cats=all_supercategories):
    '''Return all supercategories for which any of aggregated (i.e. with the
    subcategories' words as well) words is in the text'''
    text = year_key2text[(year, cik, gvkey)]
    answer = []
    for scat in cats:
        if scat2regexp[scat].search(text):
            answer.append(scat)
    return answer


load_or_compute_results('any_match')
load_or_compute_results('any_aggregated_match')
results['colors'].append('maroon')
results['colors'].append('darkred')

def all_must_match(year, cik, gvkey, cats=all_categories):
    '''Return all categories for which all descriptive words are in the text'''
    text = year_key2text[(year, cik, gvkey)]
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


load_or_compute_results('all_must_match')
results['colors'].append('red')


def n_most_matching(year, cik, gvkey, n, cats=all_categories):
    '''Return the n categories with the highest percentage of matching words'''
    text = year_key2text[(year, cik, gvkey)]
    matching_prop = {}  # \in [0;1], the proportion of descriptive words in the text
    for cat in cats:
        matching_prop[cat] = 0
        for word in d_words[cat]:
            if word in text:
                matching_prop[cat] += 1
        matching_prop[cat] /= len(d_words[cat])
    sorted_cats = sorted(list(matching_prop.keys()), key=lambda k: matching_prop[k], reverse=True)
    return sorted_cats[0:n]


def most_matching(year, cik, gvkey, cats=all_categories):
    return n_most_matching(year, cik, gvkey, 1, cats=cats)


def most_2_matching(year, cik, gvkey, cats=all_categories):
    return n_most_matching(year, cik, gvkey, 2, cats=cats)


def most_5_matching(year, cik, gvkey, cats=all_categories):
    return n_most_matching(year, cik, gvkey, 5, cats=cats)


def most_20_matching(year, cik, gvkey, cats=all_categories):
    return n_most_matching(year, cik, gvkey, 20, cats=cats)


def most_100_matching(year, cik, gvkey, cats=all_categories):
    return n_most_matching(year, cik, gvkey, 100, cats=cats)


load_or_compute_results('most_matching')
load_or_compute_results('most_2_matching')
load_or_compute_results('most_5_matching')
load_or_compute_results('most_20_matching')
load_or_compute_results('most_100_matching')
results['colors'].append('lightpink')
results['colors'].append('pink')
results['colors'].append('hotpink')
results['colors'].append('deeppink')
results['colors'].append('fuchsia')


def match_within_rank_k(year, cik, gvkey, k, cats=all_categories):
    '''Return the categories for which any descriptive word has rank <= k in the company's description'''
    text = year_key2text[(year, cik, gvkey)]
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


def match_within_rank_1(year, cik, gvkey, cats=all_categories):
    return match_within_rank_k(year, cik, gvkey, 1, cats=cats)


def match_within_rank_10(year, cik, gvkey, cats=all_categories):
    return match_within_rank_k(year, cik, gvkey, 10, cats=cats)


def match_within_rank_30(year, cik, gvkey, cats=all_categories):
    return match_within_rank_k(year, cik, gvkey, 30, cats=cats)


def match_within_rank_100(year, cik, gvkey, cats=all_categories):
    return match_within_rank_k(year, cik, gvkey, 100, cats=cats)


load_or_compute_results('match_within_rank_1')
load_or_compute_results('match_within_rank_10')
load_or_compute_results('match_within_rank_30')
load_or_compute_results('match_within_rank_100')
results['colors'].append('darkslateblue')
results['colors'].append('slateblue')
results['colors'].append('blue')
results['colors'].append('darkblue')

def alpha_most_matching(year, cik, gvkey, alpha, cats=all_categories):
    '''Return the alpha*N categories with the highest percentage of matching words where N is the number of
    categories that present at list one matching word'''
    text = year_key2text[(year, cik, gvkey)]
    matching_prop = {}  # \in [0;1], the proportion of descriptive words in the text
    for cat in cats:
        matching_prop[cat] = 0
        for word in d_words[cat]:
            if word in text:
                matching_prop[cat] += 1
        matching_prop[cat] /= len(d_words[cat])
    sorted_cats = sorted(list(matching_prop.keys()), key=lambda k: matching_prop[k], reverse=True)
    sorted_cats = [c for c in sorted_cats if matching_prop[c] > 0]
    return sorted_cats[0:int(alpha*len(sorted_cats))]


def half_most_matching(year, cik, gvkey, cats=all_categories):
    return alpha_most_matching(year, cik, gvkey, 0.5, cats=cats)


def ten_percent_most_matching(year, cik, gvkey, cats=all_categories):
    return alpha_most_matching(year, cik, gvkey, 0.1, cats=cats)


def ninety_percent_most_matching(year, cik, gvkey, cats=all_categories):
    return alpha_most_matching(year, cik, gvkey, 0.9, cats=cats)


load_or_compute_results('half_most_matching')
load_or_compute_results('ten_percent_most_matching')
load_or_compute_results('ninety_percent_most_matching')
results['colors'].append('gold')
results['colors'].append('yellow')
results['colors'].append('palegoldenrod')
