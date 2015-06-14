import pandas as pd
from data_load import *
import re


desc_table = pd.read_excel('descriptive_words.xls')
d_words = {}
for cat in all_categories:
    row = desc_table[desc_table['SIC3'] == cat]
    d_words[cat] = row[row.columns[2:]].dropna(axis=1).loc[row.index[0]].tolist()


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


results = read_pickle('results', alt={}, prefix='Exp7/')
if not 'any_match/ycik2scats' in results or not 'any_match/scat2yciks' in results:
    anymatch_ycik2scats = {}
    anymatch_scat2yciks = defaultdict(set)
    for i, (year, cik) in enumerate(all_yciks):
        print('INFO: Running strategy "Any match" on firm #{}'
              'of {}'.format(i, len(all_yciks)))
        scats = any_match(year, cik, cats=all_supercategories)
        anymatch_ycik2scats[(year, cik)] = scats
        for scat in scats:
            anymatch_scat2yciks[scat].add((year, cik))
    results['any_match/ycik2scats'] = anymatch_ycik2scats
    results['any_match/scat2yciks'] = anymatch_scat2yciks
    write_pickle('results', results, prefix='Exp7/')


