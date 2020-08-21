#!/usr/bin/env python3
# coding: utf-8
"""

A simple command line tool that predicts the sentiment of a given
passage of text. Currently only supports a shell-like interface of
taking user input and returning the results.

"""
import os
import subprocess
import warnings

from string import whitespace

import click

from joblib import load


@click.command()
@click.option(
    '--model',
    '-m',
    'model',
    default=2,
    help='int from 0-3 specifiying the model to be loaded. DEFAULT: 2'
)
@click.option(
    '--target',
    '-t',
    'target',
    default='None',
    help='a string of text for immediate analysis.'
)
def sentiment(model, target):
    """

    Analyzes the sentiment of a given document of text.
    Will return wether a given piece of text belongs
    to the three categories "negative", "neutral" and "positive".
    A probability to each class will also be returned.

    """

    cols = subprocess.run(['tput', 'cols'], text=True, capture_output=True, check=True)
    cols = int(cols.stdout.strip(whitespace))

    models_path = '../models/'
    models = sorted(models_path + mod for mod in os.listdir(models_path))

    if target != 'None':
        clf = load_model(models[model])
        results = get_results(clf, target)
        click.echo(':' + target)
        click.echo('\n'.join(results))
        return

    model_names = {
        models[0]: 'Complement Naive Bayes (bi-grams, balanced dataset)',
        models[1]: 'Complement Naive Bayes (bi-grams)',
        models[2]: 'Complement Naive Bayes (bag of words)',
        models[3]: 'Multinomial Naive Bayes (bag of words)',
    }

    intro = [
        'Welcome to SENTIMENT ANALYZER!',
        'Please input any sentence/passage for analysis.',
        '',
        'SPECIAL COMMANDS',
        '-' * cols,
        'q - terminate the program',
        '-' * cols,
        '',
        f'Loading {model_names[models[model]]} model...\n',
    ]

    intro = [string.center(cols) for string in intro]
    click.echo('\n'.join(intro))

    # Loading model using joblib
    clf = load_model(models[model])

    while True:
        inp = click.prompt('')
        if inp.strip(whitespace) == 'q':
            return
        results = get_results(clf, inp)
        click.echo('\n'.join(results))


def get_results(clf, inp):
    """

    Compute metrics and get result for prediction
    given a corpus. These currently include:
        - Sentiment (negative, neutral, positive)
        - Probability for each class

    Params
    ------

    clf - sklearn.pipeline Pipeline.
        A pipeline that should contain an transformer
        that can handle raw corpora as well as an estimator.

    inp - string.
        A string of text passed to the model.

    """
    res = ['Negative', 'Neutral', 'Positive']
    pred = clf.predict([inp]).astype(int)[0]
    prob = clf.predict_proba([inp]).ravel()
    negative, neutral, positive = ['{:.2%}'.format(p) for p in prob]

    results = [
        '',
        'Results',
        '-------',
        f'This passage is: {res[pred]}!',
        '',
        f'Probability of being Negative: {negative}',
        f'Probability of being Neutral: {neutral}',
        f'Probability of being Positive: {positive}',
        '',
    ]

    return results

def load_model(model):
    """

    Loading model using joblib.

    Params
    ------

    model - string.
        Path to model.

    Returns
    -------

    clf - python object, Pipeline
        Read from a joblib file, should be
        a sklearn Pipeline containing a transformer
        that accepts raw documents and a final estimator.

    """

    # Suppress warnings from sci-kit learn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = load(model)

    return clf
