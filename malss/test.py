# -*- coding: utf-8 -*-

from sklearn.datasets.samples_generator import make_classification,\
    make_regression
from malss import MALSS
import pandas as pd


def test_classification_2classes_small():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_classes=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               weights=[0.7, 0.3],
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'classification', n_jobs=3)
    cls.execute()
    # cls.make_report('test_classification_2classes_small')

    assert cls.algorithms[0].best_score is not None


def test_classification_multiclass_small():
    X, y = make_classification(n_samples=1000,
                               n_features=20,
                               n_classes=3,
                               n_informative=10,
                               weights=[0.6, 0.2, 0.2],
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'classification', n_jobs=3)
    cls.execute()
    # cls.make_report('test_classification_multiclass_small')

    assert cls.algorithms[0].best_score is not None


def test_classification_2classes_medium():
    X, y = make_classification(n_samples=100000,
                               n_features=10,
                               n_classes=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               weights=[0.7, 0.3],
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'classification', n_jobs=3)
    cls.execute()
    # cls.make_report('test_classification_2classes_medium')

    assert cls.algorithms[0].best_score is not None


def test_classification_2classes_big():
    X, y = make_classification(n_samples=200000,
                               n_features=20,
                               n_classes=2,
                               n_informative=3,
                               weights=[0.7, 0.3],
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'classification', n_jobs=3)
    cls.execute()
    # cls.make_report('test_classification_2classes_big')

    assert cls.algorithms[0].best_score is not None


def test_regression_small():
    X, y = make_regression(n_samples=2000,
                           n_features=10,
                           n_informative=5,
                           noise=30.0,
                           random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'regression', n_jobs=3)
    cls.execute()
    # cls.make_report('test_regression_small')

    assert cls.algorithms[0].best_score is not None


def test_regression_medium():
    X, y = make_regression(n_samples=20000,
                           n_features=10,
                           n_informative=5,
                           noise=30.0,
                           random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'regression', n_jobs=3)
    cls.execute()
    # cls.make_report('test_regression_medium')

    assert cls.algorithms[0].best_score is not None


def test_regression_big():
    X, y = make_regression(n_samples=200000,
                           n_features=10,
                           n_informative=5,
                           noise=30.0,
                           random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'regression', n_jobs=3)
    cls.execute()
    # cls.make_report('test_regression_big')

    assert cls.algorithms[0].best_score is not None


def test_classification_categorical():
    import os

    if not os.path.exists('data/heart.csv'):
        if not os.path.exists('data'):
            os.mkdir('data')

        data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv',
                           index_col=0, na_values=[''])
        data.to_csv('data/heart.csv', na_rep='')
    else:
        data = pd.read_csv('data/heart.csv', na_values=[''])

    y = data['AHD']
    del data['AHD']

    cls = MALSS(data, y, 'classification', n_jobs=3)
    cls.execute()
    # cls.make_report('test_classification_categorical')

    assert cls.algorithms[0].best_score is not None


if __name__ == "__main__":
    test_classification_categorical()
    # test_classification_2classes_small()
