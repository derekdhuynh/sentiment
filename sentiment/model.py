#!/usr/bin/env python
# coding: utf-8

import os

import json
import gzip
import h5py

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from joblib import parallel_backend, Parallel, delayed

PATH = './data/'

# Storing paths of datasets in dictionary for convenience
DATA = {
    'fashion': PATH  + 'AMAZON_FASHION_5.json.gz',
    'movies': PATH + 'Movies_and_TV_5.json.gz',
    'dictionary': PATH + 'dictionary.json',
    'words': PATH + 'words.txt',
    'cleaned': PATH + 'cleaned_amazon_reviews.json',
    'cleaned_fashion': PATH + 'cleaned_amazon_fashion_5.json',
    'amazon_data': PATH + 'amazon_reviews.hdf5'
}

def store_hdf5(file, dset_name, arr=None, readfile=None):
    """

    Either given a Numpy array or a gzip file
    containing a json file, store the contents
    into an .hdf5 file. .json file must have
    the structure of:

        {'reviewText': string, 'overall': float}

    with a new entry on each line. The overall
    represents the score on each review/text,
    scored from 0 - 5.

    Params
    ------

    file: string.
        Path of the .hdf5 file where the data
        is to be stored. Must be in the form of:

        /path/to/file.hdf5

    dset_name: string.
        Name of data set. Two datasets will
        be created if data is read from .json
        file.

    arr: array-like, any shape.
        A numpy ndarray which is to be stored
        into an .hdf5 file.

    readfile: string, default=None.
        Path of the zipped json file. Must be in
        the form of:

        /path/to/file.json.gz

    """

    with h5py.File(file, 'w') as persist:

        if arr:
            arr = np.asarray(arr)
            persist.create_dataset(dset_name, data=arr)
            return

        if readfile:
            with gzip.open(readfile, 'r') as f:
                ratings_dset = dset_name + "_ratings"
                corpus_dset = dset_name + "_review_text"
                dtype = h5py.string_dtype()
                file_size = sum((1 for line in f))
                f.seek(0) # reset cursor at start of file

                # Creating datasets for ratings and review corpora
                persist.create_dataset(
                    ratings_dset,
                    shape=(file_size, 1),
                    maxshape=(None),
                    chunks=(1000, 1),
                    dtype=np.float,
                )

                persist.create_dataset(
                    corpus_dset,
                    shape=(file_size, 1),
                    maxshape=(None),
                    chunks=(1000, 1),
                    dtype=dtype,
                )


                for ind, review in enumerate(f):
                    review = json.loads(review)

                    if 'reviewText' not in review:
                        continue

                    rating = np.asarray(review['overall']).astype(np.float)
                    corpus = np.asarray(review['reviewText'])
                    persist[ratings_dset][ind] = rating
                    persist[corpus_dset][ind] = corpus

    return f'File written at {os.path.join(os.getcwd(), file)}'

def _make_array(file):
    """
    Creating feature vectors of ratings
    annd review text from cleaned json files.
    Saving the arrays in an hdf5 file.

    """
    scores = []
    review_text = []
    with gzip.open(file) as f:
        for review in f:
            review = json.loads(review)
            if 'reviewText' not in review:
                continue
            rating = np.asarray(review['overall']).astype(np.float)
            corpus = np.asarray(review['reviewText'])
            review_text.append(corpus)
            scores.append(rating)

    scores = np.asarray(scores).astype(np.float).reshape(-1, 1)
    review_text = np.asarray(review_text).reshape(-1, 1)

    return np.core.records.fromarrays([scores, review_text], names='ratings,review_text')

def categorize_arr(arr):
    arr = np.asarray(arr)

    # Conditions to categorize reviews
    positive = (arr == 5) | (arr == 4)
    neutral = (arr == 3)
    negative = (arr == 0) | (arr == 1) | (arr == 2)

    # Setting ratings to grouped values
    arr[positive] = 2
    arr[neutral] = 1
    arr[negative] = 0

    return arr

# Preprocessing data and training model
def train_model(
        estimators, file, validate=False, folds=5, test_size=0.2, start=None, end=None):
    """
    Training sentiment analysis model using
    given estimators and a file containing
    the dataset. Also performs rudimentary
    metrics such as scoring.

    Params
    ------

    estimators: list of tuples,
        Contains the estimators' string identifier and the
        estimator itself. Used to construct a pipeline.

    file: string.
        Path of the hdf5 file.

        NOTE: datasets must be in the order of "ratings"
        and "review_text".

    validate: bool, default=False.
        If true the model is trained through cross-validation
        using the sklearn cross_validate function.

        NOTE: Not suitable for very large datasets.
        Performance implications are present.

    folds: int, default=5.
        Determines the number of fold/splits used
        in cross validation.

    test_size: float, default=0.2.
        Determines size of the test set when using the
        train_test_split function for model testing.

    Returns
    -------

    clf: sklearn.pipeline Pipeline
        Pipeline containing estimators passed from
        the estimators parameter. Mean to be saved
        to disk using joblib.

    metrics: list, shape(2)
         Contains the scoring metrics evaluated off
         of the test set. score method calculated
         from X_test and y_test, and balanced_accuracy_score
         method calculated from y_pred and y_test.

    score: dict
        Returned from cross_validate function.

    """

    with h5py.File(file, 'r') as f:
        keys = list(f.keys())
        ratings = f[keys[0]]
        corpus = f[keys[1]]
        clf = Pipeline(estimators)
        categorize = FunctionTransformer(categorize_arr)

        if validate is False:
            X_train, X_test, y_train, y_test = train_test_split(
                corpus[start: end].ravel(),
                categorize.transform(ratings[start: end].ravel()),
                test_size=test_size,
                random_state=42
            )
            with parallel_backend('threading', n_jobs=-1):
                Parallel()(delayed(clf.fit)(X_train, y_train) for i in range(1))
            # Evaluating metrics
            y_pred = clf.predict(X_test)
            score = clf.score(X_test, y_test)
            balanced_score = balanced_accuracy_score(y_test, y_pred)
            metrics = [score, balanced_score]

            return clf, metrics

        if validate is True:
            X = corpus[start: end].ravel()
            y = categorize.transform(ratings[start: end].ravel())
            score = cross_validate(clf, X, y, cv=folds, return_estimator=True, n_jobs=-1)

            return score

        return 'Something went wrong'

if __name__ == '__main__':
    pass
