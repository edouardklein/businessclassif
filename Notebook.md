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

I run this file. Results will be stored in the *confusion_matrix.pdf, *_misclassified.txt and [classif_results.csv](classif_results.csv) files.

I made a mistake : I assumed all categories had three digits. This messes up the supercategories. I'll correct that and use the file descriptive words.csv to know how much categories and supercategories we are able to recognize out of all those that exist.

I also implement a cutoff at 70% of performance (educated guess too) : if the mean performance minus the std dev is less than 70%, we don't save this classifier. If it's more than that, we say this is acceptable and use it in the second step.

I also print the coverage (number of categories for which we can train a good classifier).

Let's run this.

We can 'reliably' recognize 20 out of 83 supercategories.
We can 'reliably' recognize 11 out of 420 categories.



## Running those classifiers on yet-unknown data 

Then, we will run those classifiers on yet uncompletely classified data : the data from years after 1999. We can not fully measure success very well unless we check by hand (which would be the same as creating more labelled data), but we can at least check that the categories the companies officialy listed themselves in are correctly detected by the classifier.

I create the file [Classif_unknown.py](Classif_unknown.py) to that effect.

The results are saved in [Exp2/BUS_labels_from_classifiers.txt](Exp2/BUS_labels_from_classifiers.txt).

TODO: List, for a company that has a single label, all the labels given by the classifiers.
TODO: There exists subcategories of only two digits, put supercategories with a 0 in the end.
TODO: Seperate supercategories and categories : two separate analysis

# Exp3 Use description data

For categories that don't have a lot of labelled data, add the descriptive words from the table.

See the performance on the labelled data, and assume it will be the same on all data.

To try this, I build an IPython notebook that will let me assess the quality of any strategy : [UsingDescriptiveWords.ipynb](UsingDescriptiveWords.ipynb).

We begin by loading the data. Then we create and assess the 'perfect', 'worst', and 'random' baselines.

The plot of the score for the baselines is what one can expect it to be.

I now want to plot the score of the classifiers we learnt in Exp2. To be fair we should only compare the list we find to the list the classifier can possibly know about.

This means modifying the assessment code to restrict the testing set. Ok this is done.

I have implemented some deterministic tactics. The results are not very good. Testing new tactics will be easy thanks to the assessment code. Maybe I'll have more ideas soon.

# Misc Info
- Supercategories : first two digits
- Categories ending in 9 : misc.
- Cetegory 999 : probably mislabeled
- Category ending in 0 : ambiguous or mislabeld, or in multiple subindustries
 
