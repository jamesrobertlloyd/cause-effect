Welcome!
========

This Git repository is for members of the meetup group [Data Insights Cambridge](http://www.meetup.com/Data-Insights-Cambridge/), who are participating in the Data Insights team in the Kaggle competition ['Cause-effect pairs'](http://www.kaggle.com/c/cause-effect-pairs).

Plan
----

A Data Insights Cambridge team has been created on Kaggle. Please join this. [James Lloyd](mailto:jrl44@cam.ac.uk) has downloaded into this Git repository the competition data and any other relevant docs. 

Please work on your own separate predictions, and submit your solutions to the Git repository when ready. James will then lead the aggregation of the predictions (we might get-together for a 'hack day' to do this). James will submit final solutions to Kaggle by the deadline (in about 3 months). There will be a get-together scheduled for about 4-6 weeks time for all participants to touch-base (details will be posted on [meetup](http://www.meetup.com/Data-Insights-Cambridge/)).

Terms
-----

All proprietary solutions and data are free for all members of the repository to use under a Creative Commons Attribution 3.0 license. 

By using this Git repository you agree to these terms.

Good luck!

Repository / challenge details
==============================

See the competition [description](http://www.kaggle.com/c/cause-effect-pairs) for an introduction.

An important subtlety is the scoring metric

* The training data has 4 class labels, A->B, B->A, A|B, A-B
* The last two classes are typically referred to collectively as the 'null class'
* Final submissions to Kaggle should take the form of a number in \[-Inf, +Inf\] for each test example
* The evaluation metric is the average of two AUCs - one testing accuracy on the clasfication task A->B vs. (B->A, A|B, A-B) and the other B->A vs. (A->B, A|B, A-B)

At the moment I am undecided if it will be best for everyone to produce predicted probabilities for the 4 class problem, 3 class problem (combining A-B and A|B) or producing scores in the range \[-Inf, +Inf\] directly. For the moment I recommend trying to produce all of these measures, or work on whichever seems most natural for you. We can use everything in the final aggregation stage.

Planned workflow / repository usage
===================================

* Develop predictors using the data in data/training
* Evaluate the predictors on all training examples (data/training, data/ensemble_training, data/kaggle_validation) and put the output in the predictors directory
* Ensembling / aggregation methods will be used to combine the base predictors into a combined prediction, using data/ensemble_training as training data
* Submissions will then be made to Kaggle, recording output in submissions

Directory structure
===================

data
----

* raw - Raw data files downloaded from Kaggle
* training - Split of training data to be used for training base predictors
* ensemble_training - Split of training data to be used to train ensembling / aggregation methods
* kaggle_validation - Test data examples with unknown targets

benchmark-code
--------------

Python implementation of benchmark provided by competition organisers

submissions
-----------

See basic_python_benchmark.csv for an example of the format for submission to Kaggle

predictors
----------

Base predictions, split by type 
