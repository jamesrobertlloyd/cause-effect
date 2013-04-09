cause-effect
============

Working code for Kaggle competition. Send emails to [James Lloyd](mailto:jrl44@cam.ac.uk) if you have any questions.

Planned workflow / repository usage
===================================

* Develop predictors using the data in data/training
* Evaluate the predictors on all training examples (data/training, data/ensemble_training, data/kaggle_validation) and put the output in predictors
* Ensembling / aggregation methods will be used to combine the base predictors into a combined prediction, using data/ensemble_training as training data
* Submissions will then be made to Kaggle, recording output in submissions

Directory structure
===================

data
====

* raw - Raw data files downloaded from Kaggle
* training - Split of training data to be used for training base predictors
* ensemble_training - Split of training data to be used to train ensembling / aggregation methods
* kaggle_validation - Test data examples with unknown targets

benchmark-code
==============

Python implementation of benchmark provided by competition organisers

submissions
===========

See basic_python_benchmark.csv for an example of the format for submission to Kaggle

predictors
==========

* probabilities - Base predictions that can be interpreted as a probability
* other - All other base predictors
