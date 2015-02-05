MAchine Learning Support System
###############################

``malss`` is a python module to facilitate machine learning tasks.
This module is written to be compatible with the `scikit-learn algorithms <http://scikit-learn.org/stable/supervised_learning.html>`_ and the other scikit-learn-compatible algorithms.

.. image:: https://travis-ci.org/canard0328/malss.svg?branch=master
    :target: https://travis-ci.org/canard0328/malss

Requirements
************

These are external packages which you will need to install before installing malss.

* python (>= 2.7, 3.x's are not supported)
* numpy (>= 1.6.1)
* scipy (>= 0.9)
* scikit-learn (>= 0.15)
* matplotlib (>= 1.1)
* pandas (>= 0.13)
* jinja2 (>= 2.6)

**Windows**

If there are no binary packages matching your Python version you might to try to install these dependencies from `Christoph Gohlke Unofficial Windows installers <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

Installation
************
::

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
