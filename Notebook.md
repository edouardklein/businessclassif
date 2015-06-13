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

# Exp4 Hierarchical matching
## Restricting ourselves to supercategories
There are too many categories, therefore I code a way to restrict the analysis to supercategories.

The results are now better. The number of false-positives is less daunting.

The strategies 'N most matching' and 'Any match within rank k' in particular are quite good for the supercategories.

It would be nice to have a grid. I refactor the plotting functions in one function. Done.

## Hierarchical matching

Assuming we can perfectly match a company to the supercategory it belongs to, can we match it to the correct subcategories ?

There is a visualization problem : There are ~100 supercategories, so that would mean ~100 plots for each strategy !

Maybe by plotting all in the same plot, it looks good anyway. Let's try.

It does not look very good, but the information is legible.

Maybe I could do a specialized function to plot that, so that there is only one hull per strategy, instead of multiple hulls plotted one over the others as it is now. I wrote the `subcat_plot()` function to this end. It does look good enough.

The results are surprisingly good. The strategy 'all_must_match' seems promising.

I suspect I should take a look again at the plot function and the keywords argument. Maybe super_plot and so on are redundant now that I can give the 'all_cat' archument to comparative_plot.
I remove all *_plot fuctions and use the =all_cat= keyword argument instead.



# Exp5 Better use of descriptive words
Some descriptive words are shared between two categories. Therefore, looking for them to select only one of those is risky. There should be a way (TF-IDF?) to select only relevant, somehow exclusive words.

In this case, a document is made of the descriptive words for the supercategory and each of its subcategories. The whole corpus is the documents for all supercategories.

We train a tf_idf transformer on that corpus.

The we compute the distance between the tfidf transform of a cik text and all supercatgories' tf-idf vectors, and choose the closest(s).

I made a mistake in the arithmetic code (int(c/10)==...) that works with supercategories and subcategories, etc. I should put that into functions. Done

I code the tfidf matcher. It works OK

It appears that solutions that return a fixed number of answer are inherently imperfect, because there is not a fixed number of categories per company.

# Exp 6 Publishable results for the Machine Learning approach
## Generalities 
This is a consolidation of Exp2 and Exp4.

I modify the file [Classif_all.py](Classif_all.py) to take into account some new directives :


* In the labelled data, firms which have a supercategory as one of the labels should be ignored (probable misclabelling)
* Firms which bear the label 999 should be ignored as well.
* We need results using recall and power
* Let's not use arbitrary thresholds (20 samples, 70%), but instead let's show the continuum of performance w.r.t. the amount of labelled data
* We should reuse the visualisation from Exp4
* Things must be saved incrementally, should the computation be halted unexpectedly.
* We need to do things hierarchically. So first we train on supercategories
* Then we assume the supercategory known and we train on category within a predefined supercategory.
* We need to plot/analyze things from one firm (how many false positives/false negatives etc.) as well as from one classifier (recall and power).

That last point may take a bit of work.

I first take a look at the file from Exp2.
I'll create a new one from that, it's [Exp6.py](Exp6.py)

I modify the code that loads [labeled_firms_single.txt](labeled_firms_single.txt) so that it ignores the firms labelled with a supercategory or 999.

I clean up the classifying code in functions, so later I can load it as a module in a notebook.

It will be difficult to get statistics w.r.t. each company, because we must make sure a company was not used in the training set.

## Assessing the quality of the classifiers before the heat death of the universe

If we loop over the firms, and train each classifier making sure we don't use the firm in the training set, the computation time will be prohibitive.

If we train the classifiers beforehand, the firms will sometimes have been in the training set, which will not give us a good overview of the true performance.

We should somehow do a leave one out on all the firms.

That will give us the recall and power, as well as the trajectory of every company when confronted with a classifier that never has seen it.

This is not simply straightforward, I add the code to Exp6.py.

Results should be saved in a way that allow to recall them :

* From a company : we want to see which classifiers said 'it belongs to my category' and which said 'it does not belong to my category'
* From a category : We want to see how many false positives and false negatives there are




# Misc Info
- Supercategories : first two digits
- Categories ending in 9 : misc.
- Cetegory 999 : probably mislabeled
- Category ending in 0 : ambiguous or mislabeld, or in multiple subindustries
 
