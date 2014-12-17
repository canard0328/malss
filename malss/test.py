# -*- coding: utf-8 -*-

from sklearn.datasets.samples_generator import make_classification,\
    make_regression
from malss import MALSS
import pandas as pd
from nose.plugins.attrib import attr
import numpy as np


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
    cls = MALSS(X, y, 'classification', n_jobs=3, lang='en')
    cls.execute()
    cls.make_report('test_classification_2classes_small')
    cls.make_sample_code()

    assert len(cls.algorithms) == 6
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
    cls.make_report('test_classification_multiclass_small')
    cls.make_sample_code()

    assert len(cls.algorithms) == 5
    assert cls.algorithms[0].best_score is not None


@attr(slow=True)
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
    cls.make_report('test_classification_2classes_medium')

    assert len(cls.algorithms) == 4
    assert cls.algorithms[0].best_score is not None


@attr(travis=True)
def test_classification_2classes_medium_travis():
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
    # cls.execute()
    # cls.make_report('test_classification_2classes_medium')

    assert len(cls.algorithms) == 4
    # assert cls.algorithms[0].best_score is not None


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
    cls.make_report('test_classification_2classes_big')

    assert len(cls.algorithms) == 1
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
    cls.make_report('test_regression_small')
    cls.make_sample_code()

    assert len(cls.algorithms) == 4
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
    cls.make_report('test_regression_medium')

    assert len(cls.algorithms) == 2
    assert cls.algorithms[0].best_score is not None


@attr(travis=True)
def test_regression_medium_travis():
    X, y = make_regression(n_samples=20000,
                           n_features=10,
                           n_informative=5,
                           noise=30.0,
                           random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'regression', n_jobs=3)
    # cls.execute()
    # cls.make_report('test_regression_medium')

    assert len(cls.algorithms) == 1
    # assert cls.algorithms[0].best_score is not None


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
    cls.make_report('test_regression_big')

    assert len(cls.algorithms) == 1
    assert cls.algorithms[0].best_score is not None


def test_classification_categorical():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv',
                       index_col=0, na_values=[''])

    y = data['AHD']
    del data['AHD']

    cls = MALSS(data, y, 'classification', n_jobs=3)
    cls.execute()
    cls.make_report('test_classification_categorical')

    assert len(cls.algorithms) == 5
    assert cls.algorithms[0].best_score is not None


def test_ndarray():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv',
                       index_col=0, na_values=[''])

    y = data['AHD']
    del data['AHD']

    cls = MALSS(np.array(data), np.array(y), 'classification', n_jobs=3)
    cls.execute()
    cls.make_report('test_ndarray')

    assert len(cls.algorithms) == 5
    assert cls.algorithms[0].best_score is not None


def test_add_algorithms():
    from sklearn.ensemble import RandomForestClassifier as RF

    X, y = make_classification(n_samples=1000,
                               n_features=20,
                               n_classes=3,
                               n_informative=10,
                               weights=[0.6, 0.2, 0.2],
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS(X, y, 'classification', n_jobs=3)
    cls.add_algorithm(RF(n_jobs=3),
                      [{'n_estimators': [10, 30, 50],
                        'max_depth': [3, 5, None],
                        'max_features': [0.3, 0.6, 'auto']}],
                      'Random Forest')
    cls.execute()
    cls.make_report('test_add_algorithms')

    assert len(cls.algorithms) == 6
    assert cls.algorithms[-1].best_score is not None


def test_remove_algorithms():
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
    cls.remove_algorithm(0)
    cls.remove_algorithm()
    cls.execute()
    cls.make_report('test_remove_algorithms')
    cls.make_sample_code()

    assert len(cls.algorithms) == 3
    assert cls.algorithms[0].best_score is not None


def test_get_algorithms():
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
    algorithms = cls.get_algorithms()

    assert algorithms[0][0] == 'Support Vector Machine (RBF Kernel)'
    assert algorithms[1][0] == 'Support Vector Machine (Linear Kernel)'
    assert algorithms[2][0] == 'Logistic Regression'
    assert algorithms[3][0] == 'Decision Tree'
    assert algorithms[4][0] == 'k-Nearest Neighbors'


if __name__ == "__main__":
    test_regression_small()
