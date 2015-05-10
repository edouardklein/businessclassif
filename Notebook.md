# Business Classification
We want to be able to classify a company into multiple business categories or super-categories by analyzing a textual description  of that company.

# Exp1
The goal of the first experiment was to assess the capabilities of basic textual classification. Please see [the appropriate IPython notebook file](Business classification.ipynb). The results are that for categories with a suitable number of labelled samples, one can get very good classification results (96% accuracy with the best category (category 602)).

In category 602 there was 35 false positives. That is to say 35 companies that the classifier recognized as belonging to category 602, but that were not listed as such in the labelled data. It is possible that those companies actually belong to category 602.

The code in [Classif1.py](Classif1.py) now saves the misclassified descriptions in a [txt file](602_misclassified.txt). This file should be examined by hand to see if those companies belonged in the category they've been classified into.


# Exp2
## Training one classifier per category

We will now train a classifier for every category and supercatgory for which we have enough data, and save the classifiers in files that we can load later, and also save the results of the evaluation (Accuracy in %, as well as the confusion matrix).

I create the file [Classif_all.py](Classif_all.py) from [Classif1.py](Classif1.py). Only now, C will not be hardcoded but will span all categories and supercategories for which we have enough data (educated-guess cutoff at 20 labelled samples).

I run this file. Results will be stored in the *confusion+matrix.pdf, *_misclassified.txt and [classif_results.csv](classif_results) files.



Then, we will run those classifiers on yet uncompletely classified data : the data from years after 1998. We can not fully measure success very well unless we check by hand (which would be the same as creating more labelled data), but we can at least check that the categories the companies listed are correctly detected by the classifier.




# Misc Info
- Supercategories : first two digits
- Categories ending in 9 : misc.
- Cetegory 999 : probably mislabeled
- Category ending in 0 : ambiguous or mislabeld, or in multiple subindustries
 
