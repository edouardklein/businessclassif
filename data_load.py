'''Load the textual data in dictionaries.

 - year_cik2categories maps (year, cik) tuples to the set of categories this tuple belongs to.
 - year_cik2text maps (year, cik) tuples to the text describing this tuple
 - category2years_ciks maps a category to the set of (year, cik) that belongs to it.
 - supercategory2years_ciks maps a supercategory to the set of (year, cik) that belongs to it.
 - year_cik2supercategories maps (year, cik) tuples to the set of supercategories this tuple belongs to.
'''
import glob
import pickle
import pandas as pd
import functools
from collections import defaultdict


def read_pickle(fname, alt=None, prefix='data/'):
    '''Read a pickled variable from a file, returns the non None provided alt if
    not found, raise an exception if not found and alt is None'''
    try:
         with open(prefix+fname+'.pickle', 'rb') as f:
             return pickle.load(f)
    except FileNotFoundError:
        if not alt is None:
            return alt
        raise


def write_pickle(fname, var, prefix='data/'):
    '''Serialize var into the file named fname'''
    with open(prefix+fname+'.pickle', 'wb') as f:
        pickle.dump(var, f)


def supercat(c):
    '''Return the supercategory for category c'''
    return int(c/10)*10


def is_super(cat):
    '''Return True if cat is a supercategory'''
    return cat % 10 == 0


def should_delete(category):
    '''Whether a company should be deleted from the labelled data'''
    return is_super(category) or category == 999


def load_labeled_file():
    '''Load the labeled_firms_single.txt file'''
    with open('labeled_firms_single.txt', 'r') as f:
        lines = f.read().split('\n')[1:]
    answer = defaultdict(set)
    for l in [l for l in lines if l]:
        _, str_cik ,_, str_year, str_category = l.split('\t')
        category = int(str_category[2:-2])
        assert(category in all_categories)
        year = int(str_year)
        cik = int(str_cik)
        answer[(year, cik)].add(category)
    return answer


@functools.lru_cache()
def ycik2text(year, cik):
    '''Return the text of (year, cik) and raise a FileNotFoundError if no such text
    exists'''
    files = glob.glob('{}/{}-*'.format(year,cik))
    if not files:
        print('WARNING: file for cik:{}, year:{} not found !'.format(cik, year))
        raise FileNotFoundError
    if len(files) > 1:
        print('WARNING: more than one file for '
              'cik:{}, year:{} !'.format(cik, year))
        raise FileNotFoundError
    return open(files[0], 'r').read()


def load_texts(ycik2cats):
    '''Load the numerous text files in a dictionary'''
    year_cik2text = {}
    for i, (year, cik) in enumerate(ycik2cats):
        print('DEBUG: Loading text for firm #{} of'
              ' {}...'.format(i, len(ycik2cats)))
        try:
            year_cik2text[(year, cik)] = ycik2text(year, cik)
        except FileNotFoundError:
            pass
    write_pickle('year_cik2text', year_cik2text)
    return year_cik2text


def build_ycik2whatev():
    '''Create both year_cik2* dictionaries'''
    year_cik2categories = load_labeled_file()
    # DEBUG REMOVE
    #year_cik2categories = {k:year_cik2categories[k] for k in year_cik2categories if any([c in [283, 737, 603] for c in year_cik2categories[k]])}
    #END DEBUG
    # Stripping the mislabbelled data
    old_length = len(year_cik2categories)
    year_cik2categories = {k:year_cik2categories[k] for k in year_cik2categories
                           if not any([should_delete(cat) for
                                       cat in year_cik2categories[k]])}
    new_length = len(year_cik2categories)
    # Loading the texts
    print('INFO: We removed {} firms because one of their labels was a supercategory'
          ' or 999, now we have {} firms'.format(old_length - new_length, new_length))
    year_cik2text = load_texts(year_cik2categories)
    # Stripping the missing texts
    old_length = new_length
    year_cik2categories = {k:year_cik2categories[k] for k in year_cik2categories if
                           k in year_cik2text}
    new_length = len(year_cik2categories)
    print('INFO: We removed {} firms because we couldnt find their text,'
          ' now we have {} firms'.format(old_length - new_length, new_length))
    write_pickle('year_cik2categories', year_cik2categories)
    return year_cik2categories, year_cik2text


def build_cat2yciks(ycik2cats):
    '''Invert the given dict'''
    category2years_ciks = defaultdict(set)
    for (year, cik), cats in ycik2cats.items():
        for cat in cats:
            category2years_ciks[cat].add((year, cik))
    write_pickle('category2years_ciks', category2years_ciks)
    return category2years_ciks


def build_ycik2scats(ycik2cats):
    '''Build the mapping from a (year, cik) tuple to a set of supercategories'''
    supercats = lambda cats: set(map(supercat, cats))
    return {k:supercats(ycik2cats[k]) for k in ycik2cats}


def build_scat2yciks(cat2year_ciks):
    '''Build the mapping from a supercategory to a set of (year, cik) tuples'''
    supercategory2years_ciks = defaultdict(set)
    for cat, yciks in cat2year_ciks.items():
        for ycik in yciks:
            supercategory2years_ciks[supercat(cat)].add(ycik)
    return supercategory2years_ciks


def build_dicts():
    '''Load or, if necessary, build the different dictionaries provided by the module'''
    try:
        year_cik2categories = read_pickle('year_cik2categories')
        year_cik2text = read_pickle('year_cik2text')
    except FileNotFoundError:
        year_cik2categories, year_cik2text = build_ycik2whatev()
    try:
        category2years_ciks = read_pickle('category2years_ciks')
    except FileNotFoundError:
        category2years_ciks = build_cat2yciks(year_cik2categories)
    year_cik2supercategories = build_ycik2scats(year_cik2categories)
    supercategory2years_ciks = build_scat2yciks(category2years_ciks)
    return (year_cik2categories, year_cik2text, category2years_ciks,
            year_cik2supercategories, supercategory2years_ciks)


all_categories = list(pd.read_excel('descriptive_words.xls')['SIC3'])  # All categories that exist
all_supercategories = set(map(supercat, all_categories))  # All supercategories that exist
year_cik2categories, year_cik2text, category2years_ciks, year_cik2supercategories, supercategory2years_ciks = build_dicts()
all_yciks = set(year_cik2text.keys())

