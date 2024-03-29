MAchine Learning Support System
###############################

``malss`` is a python module to facilitate machine learning tasks.
This module is written to be compatible with the `scikit-learn algorithms <http://scikit-learn.org/stable/supervised_learning.html>`_ and the other scikit-learn-compatible algorithms.

.. image:: https://travis-ci.org/canard0328/malss.svg?branch=master
    :target: https://travis-ci.org/canard0328/malss

Dependencies
************

malss requires:

* python (>= 3.9)
* numpy (>= 1.21.2)
* scipy (>= 1.7.1)
* scikit-learn (>= 1.1.1)
* matplotlib (>= 3.4.3)
* pandas (>= 1.3.3)
* jinja2 (>= 3.1.2)

.. * PyQt5 (== 5.10) (only for interactive mode)

All modules except PyQt5 are automatically installed when installing malss.

Installation
************

  pip install malss

For interactive mode, you need to install PyQt5 using pip.

  pip install PyQt5

Example
*******

Supervised learning
===================

Classification:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  iris = load_iris()
  model = MALSS(task='classification', lang='en')
  model.fit(iris.data, iris.target, 'classification_result')
  model.generate_module_sample('classification_module_sample.py')

Regression:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_boston
  boston = load_boston()
  model = MALSS(task='regression', lang='en')
  model.fit(boston.data, boston.target, 'regression_result')
  model.generate_module_sample('regression_module_sample.py')

Change algorithm:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  from sklearn.ensemble import RandomForestClassifier as RF
  iris = load_iris()
  model = MALSS(task='classification', lang='en')
  model.fit(iris.data, iris.target, algorithm_selection_only=True)
  algorithms = model.get_algorithms()
  # check algorithms here
  model.remove_algorithm(0)  # remove the first algorithm
  # add random forest classifier
  model.add_algorithm(RF(n_jobs=3),
                    [{'n_estimators': [10, 30, 50],
                      'max_depth': [3, 5, None],
                      'max_features': [0.3, 0.6, 'auto']}],
                    'Random Forest')
  model.fit(iris.data, iris.target, 'classification_result')
  model.generate_module_sample('classification_module_sample.py')

Feature selection:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import make_friedman1
  X, y = make_friedman1(n_samples=1000, n_features=20, noise=0.0, random_state=0)
  model = MALSS(task='regression', lang='en')
  model.fit(X, y, dname='default')
  # check the analysis report
  model.select_features()
  model.fit(X, y, dname='feature_selection')
  # You can set the original data after feature selection
  # (You do not need to select features by yourself.)

.. 
  Interactive mode:

  In the interactive mode, you can interactively analyze data through a GUI.

  .. code-block:: python

    from malss import MALSS

    MALSS(lang='en', interactive=True)


Unsupervised learning
=====================

Clustering:

.. code-block:: python

  from malss import MALSS
  from sklearn.datasets import load_iris
  
  iris = load_iris()
  model = MALSS(task='clustering', lang='en')
  model.fit(iris.data, None, 'clustering_result')
  pred_dict = model.predict(iris.data)
