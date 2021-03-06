# ml_dataset_loader
[![Build Status](https://travis-ci.org/RAMitchell/ml_dataset_loader.svg?branch=master)](https://travis-ci.org/RAMitchell/ml_dataset_loader)

Provides streamlined python functions for loading machine learning datasets. Functions provided
typically (but not strictly) return a preprocessed dataset in the sklearn standard form of X,y.

Caches function calls so full datasets are only fetched once. Also provides the num_rows
parameter for each function. This allows prototyping experiments quickly using very small numbers
 of training instances.

This module is designed to be used as a submodule for a module containing machine learning
experiments.

## Adding a dataset
Add a function to datasets.py of the form:
```python
@mem.cache
def get_my_dataset(num_rows=None):
    """
    Source/description of the dataset
    Any special requirements e.g. kaggle api

    - Dimensions: xxM rows, xx columns.
    - Task: e.g. Binary classification

    :param num_rows:
    :return: X, y
    """
    # Keep imports at the local level
    import special_requirements

    #Fetch/preprocess dataset here

    return X, y

```
Add a unit test to test_datasets.py of the form:
```python
def test_my_dataset():
    # Primarily we check the dataset source is still valid
    assert url_is_alive(datasets.get_my_dataset_url)
    # Optional other tests. Do not unit test large file downloads. Travis CI will crash :)
```
Update this readme to document your new function:
```sh
# Install markdown generator if necessary
pip install pydoc-markdown
# Automatically update readme documentation
sh update_readme_documentation.sh
```
# Documentation
[comment]: # (Begin generated documentation)
# datasets
Module for loading preprocessed datasets for machine learning problems
## get_airline
```python
get_airline(num_rows=None)
```
Memoized version of get_airline(num_rows=None)

Airline dataset (http://kt.ijs.si/elena_ikonomovska/data.html)

Has categorical columns converted to ordinal and target variable "Arrival Delay" converted
to binary target.

- Dimensions: 115M rows, 13 columns.
- Task: Binary classification

:param num_rows:
:return: X, y

## get_higgs
```python
get_higgs(num_rows=None)
```
Memoized version of get_higgs(num_rows=None)

Higgs dataset from UCI machine learning repository (
https://archive.ics.uci.edu/ml/datasets/HIGGS).

- Dimensions: 11M rows, 28 columns.
- Task: Binary classification

:param num_rows:
:return: X, y

## get_cover_type
```python
get_cover_type(num_rows=None)
```
Memoized version of get_cover_type(num_rows=None)

Cover type dataset from UCI machine learning repository (
https://archive.ics.uci.edu/ml/datasets/covertype).

y contains 7 unique class labels from 1 to 7 inclusive.

- Dimensions: 581012 rows, 54 columns.
- Task: Multiclass classification

:param num_rows:
:return: X, y

## get_synthetic_classification
```python
get_synthetic_classification(num_rows=None)
```
Memoized version of get_synthetic_classification(num_rows=None)

Synthetic classification generator from sklearn (
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

- Dimensions: 10000000 rows, 100 columns.
- Task: Binary classification

:param num_rows:
:return: X, y

## get_synthetic_regression
```python
get_synthetic_regression(num_rows=None)
```
Memoized version of get_synthetic_regression(num_rows=None)

Synthetic regression generator from sklearn (
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html).

- Dimensions: 10000000 rows, 100 columns.
- Task: Regression

:param num_rows:
:return: X, y

## get_year
```python
get_year(num_rows=None)
```
Memoized version of get_year(num_rows=None)

YearPredictionMSD dataset from UCI repository (
https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)

- Dimensions: 515345 rows, 90 columns.
- Task: Regression

:param num_rows:
:return: X,y

## get_url
```python
get_url(num_rows=None)
```
Memoized version of get_url(num_rows=None)

URL reputation dataset from UCI repository (
https://archive.ics.uci.edu/ml/datasets/URL+Reputation)

Extremely sparse classification dataset. X is returned as a scipy sparse matrix.

- Dimensions: 2396130 rows, 3231961 columns.
- Task: Classification

:param num_rows:
:return: X,y

## get_bosch
```python
get_bosch(num_rows=None)
```
Memoized version of get_bosch(num_rows=None)

Bosch Production Line Performance data set (
https://www.kaggle.com/c/bosch-production-line-performance)

Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)

Contains missing values as NaN.

- Dimensions: 1.184M rows, 968 columns
- Task: Binary classification

:param num_rows:
:return: X,y

## get_adult
```python
get_adult(num_rows=None)
```
Memoized version of get_adult(num_rows=None)

Adult dataset from UCI repository (
https://archive.ics.uci.edu/ml/datasets/Adult)
Concatenates the test set to the end of the train set.
Categoricals are one hot encoded.

- Dimensions: 48842 rows, 107 columns.
- Task: Classification

:param num_rows:
:return: X,y

## get_wine_quality
```python
get_wine_quality(num_rows=None)
```
Memoized version of get_wine_quality(num_rows=None)

Wine Quality dataset from UCI repository (
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
Using the white wine data set, not the red.

- Dimensions: 4898 rows, 12 columns.
- Task: Regression

:param num_rows:
:return: X,y

## get_oshumed
```python
get_oshumed(num_rows=None)
```
Memoized version of get_oshumed(num_rows=None)

OHSUMED ranking dataset from LETOR 3.0
https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/

- Dimensions: 16140 rows, 45 columns.
- Task: Ranking

:param num_rows:
:return: X,y,query_ids

## get_epsilon
```python
get_epsilon(num_rows=None)
```

Epsilon dataset
Source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
Yuan, G. X., Ho, C. H., & Lin, C. J. (2012). An improved glmnet for l1-regularized logistic regression.
Journal of Machine Learning Research, 13(Jun), 1999-2030.

Note: This dataset contains an existing test/train split where scaling factors are calculated on the training and
applied to both the training set and the tests set. The first 400K rows are from the training set and the last 100K
rows are from the test set.

- Dimensions: 500K rows, 2K columns.
- Task: Classification

:param num_rows:
:return: X,y

[comment]: # (End generated documentation)
