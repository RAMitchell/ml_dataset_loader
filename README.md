# ml-dataset-loader
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
```
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
```
def test_my_dataset():
    # Primarily we check the dataset source is still valid
    assert url_is_alive(datasets.get_my_dataset_url)
    # Optional other tests. Do not unit test large file downloads. Travis CI will crash :)
```
Update this readme to document your new function:
```
sh update_readme_documentation.sh
```
# Documentation
[comment]: # (Begin generated documentation)
[comment]: # (End generated documentation)
