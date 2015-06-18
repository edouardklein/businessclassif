'''Load the textual data in dictionaries.

 - year_key2categories maps (year, cik, gvkey) tuples to the set of categories this tuple belongs to.
 - year_key2text maps (year, cik, gvkey) tuples to the text describing this tuple
 - category2years_keys maps a category to the set of (year, cik, gvkey) that belongs to it.
 - supercategory2years_keys maps a supercategory to the set of (year, cik, gvkey) that belongs to it.
 - year_key2supercategories maps (year, cik, gvkey) tuples to the set of supercategories this tuple belongs to.
'''
import glob
import pickle
import pandas as pd
import functools
from collections import defaultdict
import math

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


def load_labeled_files(cik2gvkey, gvkey2cik):
    '''Load the labeled_firms9(8|9).txt files'''
    answer = defaultdict(set)
    for year in ['98', '99']:
        fname = 'labeled_firms'+year+'.txt'
        year = 1900+int(year)
        with open(fname, 'r') as f:
            lines = f.read().split('\n')[1:]
        for l in [l.strip() for l in lines if l]:
            str_cik , str_gvkey, str_category = l.split('\t')
            categories = set(map(int, eval(str_category)))
            assert all([(c in all_categories) for c in categories])
            cik = int(str_cik)
            gvkey = int(str_gvkey)
            assert cik2gvkey[cik] == gvkey
            assert gvkey2cik[gvkey] == cik
            answer[(year, cik, gvkey)] |= categories
    return answer


@functools.lru_cache()
def yk2text(year, cik, gvkey):
    '''Return the text of (year, cik, gvkey) and raise a FileNotFoundError if no such text
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


def load_texts(yk2cats):
    '''Load the numerous text files in a dictionary'''
    year_key2text = {}
    for i, (year, cik, gvkey) in enumerate(yk2cats):
        print('DEBUG: Loading text for firm #{} of'
              ' {}...'.format(i, len(yk2cats)))
        try:
            year_key2text[(year, cik, gvkey)] = yk2text(year, cik, gvkey)
        except FileNotFoundError:
            pass
    write_pickle('year_key2text', year_key2text)
    return year_key2text


def build_yk2whatev(cik2gvkey, gvkey2cik):
    '''Create both year_key2* dictionaries'''
    year_key2categories = load_labeled_files(cik2gvkey, gvkey2cik)
    # DEBUG REMOVE
    #year_key2categories = {k:year_key2categories[k] for k in year_key2categories if any([c in [283, 737, 603] for c in year_key2categories[k]])}
    #END DEBUG
    # Stripping the mislabbelled data
    old_length = len(year_key2categories)
    year_key2categories = {k:year_key2categories[k] for k in year_key2categories
                           if not any([should_delete(cat) for
                                       cat in year_key2categories[k]])}
    new_length = len(year_key2categories)
    # Loading the texts
    print('INFO: We removed {} firms because one of their labels was a supercategory'
          ' or 999, now we have {} firms'.format(old_length - new_length, new_length))
    year_key2text = load_texts(year_key2categories)
    # Stripping the missing texts
    old_length = new_length
    year_key2categories = {k:year_key2categories[k] for k in year_key2categories if
                           k in year_key2text}
    new_length = len(year_key2categories)
    print('INFO: We removed {} firms because we couldnt find their text,'
          ' now we have {} firms'.format(old_length - new_length, new_length))
    write_pickle('year_key2categories', year_key2categories)
    return year_key2categories, year_key2text


def build_cat2yks(yk2cats):
    '''Invert the given dict'''
    category2years_keys = defaultdict(set)
    for (year, cik, gvkey), cats in yk2cats.items():
        for cat in cats:
            category2years_keys[cat].add((year, cik, gvkey))
    write_pickle('category2years_keys', category2years_keys)
    return category2years_keys


def build_yk2scats(yk2cats):
    '''Build the mapping from a (year, cik, gvkey) tuple to a set of supercategories'''
    supercats = lambda cats: set(map(supercat, cats))
    return {k:supercats(yk2cats[k]) for k in yk2cats}


def build_scat2yks(cat2year_keys):
    '''Build the mapping from a supercategory to a set of (year, cik, gvkey) tuples'''
    supercategory2years_keys = defaultdict(set)
    for cat, yks in cat2year_keys.items():
        for yk in yks:
            supercategory2years_keys[supercat(cat)].add(yk)
    return supercategory2years_keys


def build_cik_gvkey():
    '''Build the dicts that map cik <-> gvkey from the Excle file'''
    t = pd.read_excel('CIK_GVKEY.xlsx')
    cik2gvkey = {t['cik'][i]:(int(t['gvkey'][i]) if not math.isnan(t['gvkey'][i]) else 0) for i in t.index}
    gvkey2cik = {(int(t['gvkey'][i]) if not math.isnan(t['gvkey'][i]) else 0):t['cik'][i] for i in t.index}
    del gvkey2cik[0]
    write_pickle('cik2gvkey', cik2gvkey)
    write_pickle('gvkey2cik', gvkey2cik)
    return cik2gvkey, gvkey2cik


def build_dicts():
    '''Load or, if necessary, build the different dictionaries provided by the module'''
    try:
        cik2gvkey = read_pickle('cik2gvkey')
        gvkey2cik = read_pickle('gvkey2cik')
    except FileNotFoundError:
        cik2gvkey, gvkey2cik = build_cik_gvkey()
    try:
        year_key2categories = read_pickle('year_key2categories')
        year_key2text = read_pickle('year_key2text')
    except FileNotFoundError:
        year_key2categories, year_key2text = build_yk2whatev(cik2gvkey, gvkey2cik)
    try:
        category2years_keys = read_pickle('category2years_keys')
    except FileNotFoundError:
        category2years_keys = build_cat2yks(year_key2categories)
    year_key2supercategories = build_yk2scats(year_key2categories)
    supercategory2years_keys = build_scat2yks(category2years_keys)
    return (year_key2categories, year_key2text, category2years_keys,
            year_key2supercategories, supercategory2years_keys, cik2gvkey, gvkey2cik)


all_categories = set(pd.read_excel('descriptive_words.xls')['SIC3'])  # All categories that exist
all_supercategories = set(map(supercat, all_categories))  # All supercategories that exist
year_key2categories, year_key2text, category2years_keys, year_key2supercategories, supercategory2years_keys, cik2gvkey, gvkey2cik = build_dicts()
all_yks = set(year_key2text.keys())
known_categories = set(category2years_keys.keys())  # Categories for which we know at least one sample
unknown_categories = all_categories - known_categories  # Cetegories for which we have absolutely no samples
known_supercategories = set(map(supercat, known_categories))
unknown_supercategories = all_supercategories - known_supercategories

