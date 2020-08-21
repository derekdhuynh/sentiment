#!/usr/bin/env python3
# coding: utf-8
"""

A simple command line tool that predicts the sentiment of a given
passage of text. Currently only supports a shell-like interface of
taking user input and returning the results.

NOTES:
------
- Special commands right now are wonky, figure out how to refactor
    and make the way I'm doing it right now work
- Worst case, make every command an if check.

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
@click.option(
    '--file',
    '-f',
    'file',
    default='None',
    help='path to a text file'
)
def sentiment(model, target, file):
    """

    Analyzes the sentiment of a given document of text.
    Will return wether a given piece of text belongs
    to the three categories "negative", "neutral" and "positive".
    A probability to each class will also be returned.

    """

    # Getting number of columns in terminal.
    cols = subprocess.run(['tput', 'cols'], text=True, capture_output=True, check=True)
    cols = int(cols.stdout.strip(whitespace))

    # Getting paths to models and sorting them.
    models_path = '../models/'
    models = sorted(models_path + mod for mod in os.listdir(models_path))

    model_names = {
        models[0]: 'Complement Naive Bayes (bi-grams, balanced dataset)',
        models[1]: 'Complement Naive Bayes (bi-grams)',
        models[2]: 'Complement Naive Bayes (bag of words)',
        models[3]: 'Multinomial Naive Bayes (bag of words)',
    }

    # Simply evaluate sentiment and write to stdout
    # If target string is given.
    if target != 'None':
        clf = load_model(models[model])
        results = get_results(clf, target)
        click.echo(':' + target)
        click.echo('\n'.join(results))
        return

    intro = [
        'Welcome to SENTIMENT ANALYZER!',
        'Please input any sentence/passage for analysis.',
        '',
        'SPECIAL COMMANDS',
        '-' * cols,
        'q - terminate the program',
        'm - select model',
        '-' * cols,
        '',
        f'Loading {model_names[models[model]]} model...\n',
    ]

    intro = [string.center(cols) for string in intro]
    click.echo('\n'.join(intro))

    # Loading model using joblib
    clf = load_model(models[model])

    # Sentiment analysis given file option.
    if file != 'None':
        with open(file, 'r') as text:
            contents = text.read()
            results = get_results(clf, contents)
            click.echo('\n'.join(results))

    # Mapping commands to function refs.
    commands = {
        'q': exit,
        'm': change_model
    }

    while True:
        inp = click.prompt('')
        stripped = inp.strip(whitespace)
        if stripped in commands:
            model = commands[stripped]() # Will simply exit if q
            click.echo(f'Loading {model_names[models[model]]} model...\n')
            clf = load_model(models[model])
            continue
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

    clf: sklearn.pipeline Pipeline.
        A pipeline that should contain an transformer
        that can handle raw corpora as well as an estimator.

    inp: string.
        A string of text passed to the model.

    Returns
    -------

    results: list of strings.
        Results displayed to stdout. Echoed with '\n'.join(results)

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

    model: string.
        Path to model.

    Returns
    -------

    clf: python object, Pipeline
        Read from a joblib file, should be
        a sklearn Pipeline containing a transformer
        that accepts raw documents and a final estimator.

    """

    # Suppress warnings from sci-kit learn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = load(model)

    return clf

def change_model():
    """

    Model selection screen.

    """

    select = [
        '',
        'Please select a model:',
        '----------------------',
        '0 - Complement Naive Bayes (bi-grams, balanced dataset)',
        '1 - Complement Naive Bayes (bi-grams)',
        '2 - Complement Naive Bayes (bag of words)',
        '3 - Multinomial Naive Bayes (bag of words)',
        '',
    ]

    click.echo('\n'.join(select))
    while True:
        inp = click.prompt('')
        try:
            assert int(inp) >= 0 and int(inp) <= 3
        except (AssertionError, ValueError):
            click.echo('Please input a valid number')
        else:
            inp = int(inp)
            break

    return inp
