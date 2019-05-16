MAchine Learning Support System
###############################

``malss`` is a python module to facilitate machine learning tasks.
This module is written to be compatible with the `scikit-learn algorithms <http://scikit-learn.org/stable/supervised_learning.html>`_ and the other scikit-learn-compatible algorithms.

.. image:: https://travis-ci.org/canard0328/malss.svg?branch=master
    :target: https://travis-ci.org/canard0328/malss

Dependencies
************

malss requires:

* python (>= 3.6)
* numpy (>= 1.10.2)
* scipy (>= 0.16.1)
* scikit-learn (>= 0.19)
* matplotlib (>= 1.5.1)
* pandas (>= 0.14.1)
* jinja2 (>= 2.8)
* PyQt5 (>= 5.12) (only for interactive mode)

All modules except PyQt5 are automatically installed when installing malss.

Installation
************

  pip install malss

For interactive mode, you need to install PyQt5 using pip.

  pip install PyQt5

Example
*******

Classification:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  iris = load_iris()
  clf = MALSS(task='classification', lang='en')
  clf.fit(iris.data, iris.target, 'classification_result')
  clf.generate_module_sample('classification_module_sample.py')

Regression:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_boston
  boston = load_boston()
  clf = MALSS(task='regression', lang='en')
  clf.fit(boston.data, boston.target, 'regression_result')
  clf.generate_module_sample('regression_module_sample.py')

Change algorithm:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  from sklearn.ensemble import RandomForestClassifier as RF
  iris = load_iris()
  clf = MALSS(task='classification', lang='en')
  clf.fit(iris.data, iris.target, algorithm_selection_only=True)
  algorithms = clf.get_algorithms()
  # check algorithms here
  clf.remove_algorithm(0)  # remove the first algorithm
  # add random forest classifier
  clf.add_algorithm(RF(n_jobs=3),
                    [{'n_estimators': [10, 30, 50],
                      'max_depth': [3, 5, None],
                      'max_features': [0.3, 0.6, 'auto']}],
                    'Random Forest')
  clf.fit(iris.data, iris.target, 'classification_result')
  clf.generate_module_sample('classification_module_sample.py')

Interactive mode:

In the interactive mode, you can interactively analyze data through a GUI.

.. code-block:: python

  from malss import MALSS

  MALSS(lang='en', interactive=True)