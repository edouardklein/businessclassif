# To run this file:
# C:\Anaconda3\python.exe INPUTFOLDER METHODFORSUP METHDODFORCAT
'''This script runs the specified methods on the years in the source code, for the unknown categories'''
from data_load import *
import sys
from os.path import basename
from glob import glob
from Exp7 import *

input_dir = sys.argv[1]
method_super = eval(sys.argv[2])
method_sub = eval(sys.argv[3])


output = open('Exp8.csv', 'w')
output.write('filename,cik,gvkey,year,predicted\n')

for fname in glob.glob(input_dir+'/*.txt'):
    print('Finding categories for file'+fname)
    text = open(fname, 'r').read()
    cik = int(basename(fname).split('-')[0])
    year = int(basename(fname).split('-')[1])
    if year >50:
        year = 1900+year
    else:
        year = 2000+year
    gvkey = cik2gvkey[cik]
    yk = (year, cik, gvkey)
    year_key2text[yk] = text
    scats = method_super(year, cik, gvkey, cats=unknown_supercategories)
    cats = set()
    for scat in scats:
        possible_cats = [c for c in unknown_categories if supercat(c) in scats]
        _cats = method_sub(year, cik, gvkey, cats=possible_cats)
        cats |= set(_cats)
    for cat in cats:
        output.write(','.join([basename(fname), str(cik), str(gvkey), str(year), str(cat)])+'\n')
