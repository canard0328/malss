MAchine Learning Support System
###############################

``malss`` is a python module to facilitate machine learning tasks.
This module is written to be compatible with the `scikit-learn algorithms <http://scikit-learn.org/stable/supervised_learning.html>`_ and the other scikit-learn-compatible algorithms.

.. image:: https://travis-ci.org/canard0328/malss.svg?branch=master
    :target: https://travis-ci.org/canard0328/malss

Requirements
************

These are external packages which you will need to install before installing malss.

* python (>= 2.7 or >= 3.4)
* numpy (>= 1.10.2)
* scipy (>= 0.16.1)
* scikit-learn (>= 0.17)
* matplotlib (>= 1.5.1)
* pandas (>= 0.14.1)
* jinja2 (>= 2.8)

I highly recommend `Anaconda <https://www.continuum.io/downloads>`_.
Anaconda conveniently installs packages listed above.

Installation
************

Do not install package dependencies::

  pip install --no-deps malss

Install package dependencies::

  pip install malss

Example
*******

Classification:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  iris = load_iris()
  clf = MALSS('classification')
  clf.fit(iris.data, iris.target, 'classification_result')
  clf.generate_module_sample('classification_module_sample.py')

Regression:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_boston
  boston = load_boston()
  clf = MALSS('regression')
  clf.fit(boston.data, boston.target, 'regression_result')
  clf.generate_module_sample('regression_module_sample.py')

Change algorithm:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  from sklearn.ensemble import RandomForestClassifier as RF
  iris = load_iris()
  clf = MALSS('classification')
  clf.fit(iris.data, iris.target, algorithm_selection_only=True)
  algorithms = clf.get_algorithms()
  # check algorithms here
  clf.remove_algorithm(0)
  clf.add_algorithm(RF(n_jobs=3),
                    [{'n_estimators': [10, 30, 50],
                      'max_depth': [3, 5, None],
                      'max_features': [0.3, 0.6, 'auto']}],
                    'Random Forest')
  clf.fit(iris.data, iris.target, 'classification_result')
  clf.generate_module_sample('classification_module_sample.py')

API
***
View the `documentation here <https://pythonhosted.org/malss/>`_.
