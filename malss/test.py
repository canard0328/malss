# -*- coding: utf-8 -*-

from sklearn.datasets.samples_generator import make_classification,\
    make_regression
from .malss import MALSS
import pandas as pd
from nose.plugins.attrib import attr
import numpy as np


def test_classification_2classes_small():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_classes=2,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification').fit(X, y,
                                      'test_classification_2classes_small')
    cls.generate_module_sample()

    from sklearn.metrics import f1_score
    pred = cls.predict(X)
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 6
    assert cls.algorithms[0].best_score is not None


def test_classification_2classes_small_jp():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_classes=2,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification',
                lang='jp').fit(X, y, 'test_classification_2classes_small_jp')
    cls.generate_module_sample()

    from sklearn.metrics import f1_score
    pred = cls.predict(X)
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 6
    assert cls.algorithms[0].best_score is not None


def test_classification_multiclass_small():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_classes=3,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification').fit(X, y,
                                      'test_classification_multiclass_small')
    cls.generate_module_sample()

    from sklearn.metrics import f1_score
    pred = cls.predict(X)
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 6
    assert cls.algorithms[0].best_score is not None


@attr(slow=True)
def test_classification_2classes_medium():
    X, y = make_classification(n_samples=100000,
                               n_features=10,
                               n_classes=2,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification').fit(X, y,
                                      'test_classification_2classes_medium')

    from sklearn.metrics import f1_score
    pred = cls.predict(X)
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 4
    assert cls.algorithms[0].best_score is not None


def test_classification_2classes_big():
    X, y = make_classification(n_samples=200000,
                               n_features=20,
                               n_classes=2,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification').fit(X, y,
                                      'test_classification_2classes_big')
    cls.generate_module_sample()

    from sklearn.metrics import f1_score
    pred = cls.predict(X)
    print(f1_score(y, pred, average=None))

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
    cls = MALSS('regression').fit(X, y, 'test_regression_small')
    cls.generate_module_sample()

    from sklearn.metrics import mean_squared_error
    pred = cls.predict(X)
    print(mean_squared_error(y, pred))

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
    cls = MALSS('regression').fit(X, y, 'test_regression_medium')
    cls.generate_module_sample()

    from sklearn.metrics import mean_squared_error
    pred = cls.predict(X)
    print(mean_squared_error(y, pred))

    assert len(cls.algorithms) == 2
    assert cls.algorithms[0].best_score is not None


def test_regression_big():
    X, y = make_regression(n_samples=200000,
                           n_features=10,
                           n_informative=5,
                           noise=30.0,
                           random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('regression').fit(X, y, 'test_regression_big')
    cls.generate_module_sample()

    from sklearn.metrics import mean_squared_error
    pred = cls.predict(X)
    print(mean_squared_error(y, pred))

    assert len(cls.algorithms) == 1
    assert cls.algorithms[0].best_score is not None


def test_classification_categorical():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv',
                       index_col=0, na_values=[''])

    y = data['AHD']
    del data['AHD']

    cls = MALSS('classification').fit(data, y,
                                      'test_classification_categorical')
    cls.generate_module_sample()

    pred = cls.predict(data)
    from sklearn.metrics import f1_score
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 6
    assert cls.algorithms[0].best_score is not None


def test_ndarray():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv',
                       index_col=0, na_values=[''])

    y = data['AHD']
    del data['AHD']

    cls = MALSS('classification').fit(np.array(data), np.array(y),
                                      'test_ndarray')
    cls.generate_module_sample()

    from sklearn.metrics import f1_score
    pred = cls.predict(np.array(data))
    print(f1_score(y, pred, average=None))

    assert len(cls.algorithms) == 6
    assert cls.algorithms[0].best_score is not None


def test_change_algorithms():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_classes=2,
                               n_clusters_per_class=1,
                               random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    cls = MALSS('classification')
    cls.fit(X, y, algorithm_selection_only=True)
    algorithms = cls.get_algorithms()

    assert algorithms[0][0] == 'Support Vector Machine (RBF Kernel)'
    assert algorithms[1][0] == 'Random Forest'
    assert algorithms[2][0] == 'Support Vector Machine (Linear Kernel)'
    assert algorithms[3][0] == 'Logistic Regression'
    assert algorithms[4][0] == 'Decision Tree'
    assert algorithms[5][0] == 'k-Nearest Neighbors'

    cls.remove_algorithm(0)
    cls.remove_algorithm()
    algorithms = cls.get_algorithms()
    assert algorithms[0][0] == 'Random Forest'
    assert algorithms[1][0] == 'Support Vector Machine (Linear Kernel)'
    assert algorithms[2][0] == 'Logistic Regression'
    assert algorithms[3][0] == 'Decision Tree'

    from sklearn.ensemble import ExtraTreesClassifier as ET
    cls.add_algorithm(ET(n_jobs=3),
                      [{'n_estimators': [10, 30, 50],
                        'max_depth': [3, 5, None],
                        'max_features': [0.3, 0.6, 'auto']}],
                      'Extremely Randomized Trees')
    algorithms = cls.get_algorithms()
    assert algorithms[0][0] == 'Random Forest'
    assert algorithms[1][0] == 'Support Vector Machine (Linear Kernel)'
    assert algorithms[2][0] == 'Logistic Regression'
    assert algorithms[3][0] == 'Decision Tree'
    assert algorithms[4][0] == 'Extremely Randomized Trees'


if __name__ == "__main__":
    test_classification_2classes_small()
