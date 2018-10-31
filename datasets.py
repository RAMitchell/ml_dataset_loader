"""Module for loading preprocessed datasets for machine learning problems"""
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.externals.joblib.memory import Memory

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module

mem = Memory("./mycache")

get_airline_url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'


@mem.cache
def get_airline(num_rows=None):
    """
    Airline dataset (http://kt.ijs.si/elena_ikonomovska/data.html)

    Has categorical columns converted to ordinal and target variable "Arrival Delay" converted
    to binary target.

    - Dimensions: 115M rows, 13 columns.
    - Task: Binary classification

    :param num_rows:
    :return: X, y
    """
    filename = 'airline_14col.data.bz2'
    if not os.path.isfile(filename):
        urlretrieve(get_airline_url, filename)

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance":
            dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df = pd.read_csv(filename,
                     names=cols, dtype=dtype_columns, nrows=num_rows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]

    del df
    return X, y


get_higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'  # pylint: disable=line-too-long


@mem.cache
def get_higgs(num_rows=None):
    """
    Higgs dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HIGGS).

    - Dimensions: 11M rows, 28 columns.
    - Task: Binary classification

    :param num_rows:
    :return: X, y
    """
    filename = 'HIGGS.csv.gz'
    if not os.path.isfile(filename):
        urlretrieve(get_higgs_url, filename)
    higgs = pd.read_csv(filename, nrows=num_rows)
    X = higgs.iloc[:, 1:].values
    y = higgs.iloc[:, 0].values

    return X, y


@mem.cache
def get_cover_type(num_rows=None):
    """
    Cover type dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/covertype).

    y contains 7 unique class labels from 1 to 7 inclusive.

    - Dimensions: 581012 rows, 54 columns.
    - Task: Multiclass classification

    :param num_rows:
    :return: X, y
    """
    X, y = datasets.fetch_covtype(return_X_y=True)
    if num_rows is not None:
        X = X[0:num_rows]
        y = y[0:num_rows]

    return X, y


@mem.cache
def get_synthetic_classification(num_rows=None):
    """
    Synthetic classification generator from sklearn (
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

    - Dimensions: 10000000 rows, 100 columns.
    - Task: Binary classification

    :param num_rows:
    :return: X, y
    """
    if num_rows is None:
        num_rows = 10000000

    return datasets.make_classification(n_samples=num_rows, random_state=0)


@mem.cache
def get_synthetic_regression(num_rows=None):
    """
    Synthetic regression generator from sklearn (
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html).

    - Dimensions: 10000000 rows, 100 columns.
    - Task: Regression

    :param num_rows:
    :return: X, y
    """
    if num_rows is None:
        num_rows = 10000000

    return datasets.make_regression(n_samples=num_rows, bias=100, noise=1.0, random_state=0)


get_year_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'  # pylint: disable=line-too-long


@mem.cache
def get_year(num_rows=None):
    """
    YearPredictionMSD dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)

    - Dimensions: 515345 rows, 90 columns.
    - Task: Regression

    :param num_rows:
    :return: X,y
    """
    filename = 'YearPredictionMSD.txt.zip'
    if not os.path.isfile(filename):
        urlretrieve(get_year_url, filename)

    year = pd.read_csv(filename, header=None, nrows=num_rows)
    X = year.iloc[:, 1:].values
    y = year.iloc[:, 0].values
    return X, y


get_url_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'  # pylint: disable=line-too-long


@mem.cache
def get_url(num_rows=None):
    """
    URL reputation dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/URL+Reputation)

    Extremely sparse classification dataset. X is returned as a scipy sparse matrix.

    - Dimensions: 2396130 rows, 3231961 columns.
    - Task: Classification

    :param num_rows:
    :return: X,y
    """
    from scipy.sparse import vstack
    filename = 'url_svmlight.tar.gz'
    if not os.path.isfile(filename):
        urlretrieve(get_url_url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

    num_files = 120
    files = ['url_svmlight/Day{}.svm'.format(day) for day in range(num_files)]
    data = datasets.load_svmlight_files(files)
    X = vstack(data[::2])
    y = np.concatenate(data[1::2])

    y[y < 0.0] = 0.0

    if num_rows is not None:
        X = X[0:num_rows]
        y = y[0:num_rows]

    return X, y


@mem.cache
def get_bosch(num_rows=None):
    """
    Bosch Production Line Performance data set (
    https://www.kaggle.com/c/bosch-production-line-performance)

    Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)

    Contains missing values as NaN.

    - Dimensions: 1.184M rows, 968 columns
    - Task: Binary classification

    :param num_rows:
    :return: X,y
    """
    os.system("kaggle competitions download -c bosch-production-line-performance -f "
              "train_numeric.csv.zip -p .")
    X = pd.read_csv("train_numeric.csv.zip", index_col=0, compression='zip', dtype=np.float32,
                    nrows=num_rows)
    y = X.iloc[:, -1]
    X.drop(X.columns[-1], axis=1, inplace=True)
    return X, y


get_adult_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'  # pylint: disable=line-too-long
get_adult_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'  # pylint: disable=line-too-long


@mem.cache
def get_adult(num_rows=None):
    """
    Adult dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/Adult)
    Concatenates the test set to the end of the train set.
    Categoricals are one hot encoded.

    - Dimensions: 48842 rows, 107 columns.
    - Task: Classification

    :param num_rows:
    :return: X,y
    """
    train_filename = 'adult.data'
    test_filename = 'adult.test'
    if not os.path.isfile(train_filename):
        urlretrieve(get_adult_train_url, train_filename)

    if not os.path.isfile(test_filename):
        urlretrieve(get_adult_test_url, test_filename)

    adult = pd.read_csv(train_filename, header=None)
    adult = adult.append(pd.read_csv(test_filename, header=None, skiprows=1))
    y = np.where(adult.iloc[:, -1] == ' >50K', 1, 0)
    adult.drop(adult.columns[-1], axis=1, inplace=True)
    adult = pd.get_dummies(adult)
    X = adult.iloc[:, :-1].values
    if num_rows is not None:
        X = X[:num_rows]
        y = y[:num_rows]

    return X, y


get_wine_quality_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'  # pylint: disable=line-too-long


@mem.cache
def get_wine_quality(num_rows=None):
    """
    Wine Quality dataset from UCI repository (
    https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Using the white wine data set, not the red.

    - Dimensions: 4898 rows, 12 columns.
    - Task: Regression

    :param num_rows:
    :return: X,y
    """
    filename = 'winequality-white.csv'
    if not os.path.isfile(filename):
        urlretrieve(get_wine_quality_url, filename)

    wine = pd.read_csv(filename, header=0, nrows=num_rows, delimiter=";")

    X = wine.iloc[:, :-1].values
    y = wine.iloc[:, -1].values
    return X, y


get_oshumed_url = 'https://drive.google.com/uc?export=download&id=1h_r1vyJaudJba5ha096_HQiXpJmyTJbx'  # pylint: disable=line-too-long


@mem.cache
def get_oshumed(num_rows=None):
    """
    OHSUMED ranking dataset from LETOR 3.0
    https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/

    - Dimensions: 16140 rows, 45 columns.
    - Task: Ranking

    :param num_rows:
    :return: X,y,query_ids
    """
    filename = 'OHSUMED.txt'
    if not os.path.isfile(filename):
        urlretrieve(get_oshumed_url, "OHSUMED.zip")
        import zipfile
        with zipfile.ZipFile("OHSUMED.zip") as z:
            with open(filename, 'wb') as f:
                f.write(z.read("OHSUMED/Feature-min/ALL/" + filename))

    from sklearn.datasets import load_svmlight_file
    X, y, qid = load_svmlight_file("OHSUMED.txt", query_id=True)  # pylint: disable=unbalanced-tuple-unpacking
    if num_rows is not None:
        X = X[:num_rows]
        y = y[:num_rows]
        qid = qid[:num_rows]

    return X, y, qid
