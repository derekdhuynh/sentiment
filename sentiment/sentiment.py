#!/usr/bin/env python3
# coding: utf-8
"""

A simple command line tool that predicts the sentiment of a given
passage of text. Currently only supports a shell-like interface of
taking user input and returning the results.

"""
import os
import warnings
import click
from string import whitespace

from joblib import load

@click.command()
def sentiment():
    """

    Analyzes the sentiment of a given document of text.
    Will return wether a given piece of text belongs
    to the three categories "negative", "neutral" and "positive".
    A probability to each class will also be returned.

    """

    models_path = os.environ['HOME'] + '/programming/sentiment_naive/models/cnb_bow.joblib'

    intro = [
        'Welcome to SENTIMENT ANALYZER!',
        'Please input any sentence/passage for analysis.',
        '',
        'SPECIAL COMMANDS',
        '----------------',
        'q - terminate the program',
        '',
        ]

    model_name = 'Complement Naive Bayes (bag of words)'
    click.echo('\n'.join(intro))

    # Loading model from joblib
    click.echo(f'Loading {model_name} model...\n')

    # Suppress warnings from sci-kit learn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = load(models_path)

    while True:
        inp = click.prompt('')
        if inp.strip(whitespace) == 'q':
            return
        get_results(clf, inp)


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
    click.echo('Results')
    click.echo('-------')
    click.echo('This passage is: ' + res[pred] + '!\n')
    probabilities = [
        f'Probability of being Negative: {negative}',
        f'Probability of being Neutral: {neutral}',
        f'Probability of being Positive: {positive}',
    ]
    click.echo('\n'.join(probabilities))
