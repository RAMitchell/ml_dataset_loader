# ml_dataset_loader
[![Build Status](https://travis-ci.org/RAMitchell/ml-dataset-loader.svg?branch=master)](https://travis-ci.org/RAMitchell/ml-dataset-loader)

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
<h1 id="datasets">datasets</h1>

Module for loading preprocessed datasets for machine learning problems
<h2 id="datasets.get_higgs">get_higgs</h2>

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

<h2 id="datasets.get_cover_type">get_cover_type</h2>

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

<h2 id="datasets.get_synthetic_regression">get_synthetic_regression</h2>

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

<h2 id="datasets.get_year">get_year</h2>

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

<h2 id="datasets.get_url">get_url</h2>

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

[comment]: # (End generated documentation)
