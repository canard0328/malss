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
* scikit-learn (>= 0.14)
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

Classification::

  from malss import MALSS
  from sklearn.datasets import load_iris
  iris = load_iris()
  cls = MALSS(iris.data, iris.target, task='classification')
  cls.execute()
  cls.make_report('classification_result')
  cls.make_sample_code('classification_sample_code.py')

Regression::

  from malss import MALSS
  from sklearn.datasets import load_boston
  boston = load_boston()
  cls = MALSS(boston.data, boston.target, task='regression')
  cls.execute()
  cls.make_report('regression_result')
  cls.make_sample_code('regression_sample_code.py')

Change algorithm::

  from malss import MALSS
  from sklearn.datasets import load_iris
  iris = load_iris()
  cls = MALSS(iris.data, iris.target, task='classification')
  algorithms = cls.get_algorithms()
  # check algorithms here
  cls.remove_algorithm(0)
  cls.add_algorithm(RF(n_jobs=3),
                    [{'n_estimators': [10, 30, 50],
                      'max_depth': [3, 5, None],
                      'max_features': [0.3, 0.6, 'auto']}],
                    'Random Forest')
  cls.execute()
  cls.make_report('classification_result')
  cls.make_sample_code('classification_sample_code.py')

API
***
View the `documentation here <https://pythonhosted.org/malss/>`_.