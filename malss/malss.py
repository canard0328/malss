# -*- coding: utf-8 -*-

import os
import io
import warnings
import argparse
import numpy as np
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor,\
    SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning

from .algorithm import Algorithm
from .data import Data


class MALSS(object):
    def __init__(self, task=None, shuffle=True, standardize=True, scoring=None,
                 cv=5, n_jobs=-1, random_state=0, lang='en', verbose=True,
                 interactive=False):
        """
        Initialize parameters.

        Parameters
        ----------
        task : string
            Specifies the task of the analysis. It must be one of
            'classification', 'regression'.
        shuffle : boolean, optional (default=True)
            Whether to shuffle the data.
        standardize : boolean, optional (default=True)
            Whether to sdandardize the data.
        scoring : string, callable or None, optional, default: None
            A string (see scikit-learn's model evaluation documentation) or
            a scorer callable object / function with
            signature scorer(estimator, X, y).
            mean_squared_error (for regression task) or f1 (for classification
            task) is used by default.
        cv : integer or cross-validation generator.
            If an integer is passed, it is the number of folds (default 3).
            K-fold cv (for regression task) or Stratified k-fold cv
            (for classification task) is used by default.
            Specific cross-validation objects can be passed, see
            sklearn.model_selection module for the list of possible objects.
        n_jobs : integer, optional (default=-1)
            The number of jobs to run in parallel. If -1, then the number of
            jobs is set to the number of cores - 1.
        random_state : int seed, RandomState instance, or None (default=0)
            The seed of the pseudo random number generator
        lang : string (default='en')
            Specifies the language in the report. It must be one of
            'en' (English), 'jp' (Japanese).
        verbose : boolean, default: True
            Enable verbose output.
        interactive : boolean, default: False
            Run MALSS with interactive application mode.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('--lang', '-l', nargs=1, choices=['en', 'jp'])

        if interactive:
            import sys
            from .app import App
            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                print('PyQt5 is required.')
                sys.exit()
            app = QApplication(sys.argv)
            args = parser.parse_args()
            if args.lang is not None:
                lang = args.lang[0]
            App(lang=lang)
            sys.exit(app.exec_())

        self.is_ready = False

        self.shuffle = shuffle

        self.standardize = standardize

        if task is None:
            raise ValueError("Set task ('classification' or 'regression').")
        elif task == 'classification':
            self.scoring = 'f1_weighted' if scoring is None else scoring
        elif task == 'regression':
            self.scoring =\
                    'neg_mean_squared_error' if scoring is None else scoring
        else:
            raise ValueError('task:%s is not supported' % task)
        self.task = task

        self.cv = cv
        if n_jobs == -1:
            self.n_jobs = np.max([multiprocessing.cpu_count() - 1, 1])
        else:
            self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        if lang != 'en' and lang != 'jp':
            raise ValueError('lang:%s is no supported' % lang)
        self.lang = lang

        self.data = None

        self.algorithms = []

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    def __choose_algorithm(self):
        algorithms = []
        if self.task == 'classification':
            if self.data.X.shape[0] * self.data.X.shape[1] <= 1e+06:
                if self.data.X.shape[0] ** 2 * self.data.X.shape[1] <= 1e+09:
                    algorithms.append(
                        Algorithm(
                            SVC(random_state=self.random_state, kernel='rbf'),
                            [{'C': [1, 10, 100, 1000],
                              'gamma': [1e-3, 1e-2, 1e-1, 1.0]}],
                            'Support Vector Machine (RBF Kernel)',
                            ('http://scikit-learn.org/stable/modules/'
                             'generated/sklearn.svm.SVC.html')))
                    algorithms.append(
                        Algorithm(
                            RandomForestClassifier(
                                random_state=self.random_state,
                                n_estimators=500,
                                n_jobs=1),
                            [{'max_features': [0.3, 0.6, 0.9],
                              'max_depth': [3, 7, 11]}],
                            'Random Forest',
                            ('http://scikit-learn.org/stable/modules/'
                             'generated/'
                             'sklearn.ensemble.RandomForestClassifier.html')))
                algorithms.append(
                    Algorithm(
                        LinearSVC(random_state=self.random_state),
                        [{'C': [0.1, 1, 10, 100]}],
                        'Support Vector Machine (Linear Kernel)',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.svm.LinearSVC.html')))
                algorithms.append(
                    Algorithm(
                        LogisticRegression(random_state=self.random_state),
                        [{'C': [0.1, 0.3, 1, 3, 10]}],
                        'Logistic Regression',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.linear_model.LogisticRegression.html')))
                algorithms.append(
                    Algorithm(
                        DecisionTreeClassifier(random_state=self.random_state),
                        [{'max_depth': [3, 5, 7, 9, 11]}],
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeClassifier.html')))

                # Too small data doesn't suit for kNN.
                min_nn = int(
                    0.1 * (self.cv - 1) * self.data.X.shape[0] / self.cv)
                # where 0.1 means smallest data size ratio of learning_curve
                # function.
                # The value of min_nn isn't accurate when cv is stratified.
                if min_nn >= 11:
                    algorithms.append(
                        Algorithm(
                            KNeighborsClassifier(),
                            [{'n_neighbors': list(range(2, min(20, min_nn + 1),
                                                        4))}],
                            'k-Nearest Neighbors',
                            ('http://scikit-learn.org/stable/modules/'
                             'generated/sklearn.neighbors.KNeighborsClassifier'
                             '.html')))
            else:
                algorithms.append(
                    Algorithm(
                        SGDClassifier(
                            random_state=self.random_state,
                            n_jobs=1),
                        [{'alpha': [1e-05, 3e-05, 1e-04, 3e-04, 1e-03]}],
                        'SGD Classifier',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.linear_model.SGDClassifier.html')))
        if self.task == 'regression':
            if self.data.X.shape[0] * self.data.X.shape[1] <= 1e+06:
                if self.data.X.shape[0] ** 2 * self.data.X.shape[1] <= 1e+09:
                    algorithms.append(
                        Algorithm(
                            SVR(kernel='rbf'),
                            [{'C': [1, 10, 100, 1000],
                              'gamma': [1e-3, 1e-2, 1e-1, 1.0]}],
                            'Support Vector Machine (RBF Kernel)',
                            ('http://scikit-learn.org/stable/modules/'
                             'generated/sklearn.svm.SVR.html')))
                    algorithms.append(
                        Algorithm(
                            RandomForestRegressor(
                                random_state=self.random_state,
                                n_estimators=500,
                                n_jobs=1),
                            [{'max_features': [0.3, 0.6, 0.9],
                              'max_depth': [3, 7, 11]}],
                            'Random Forest',
                            ('http://scikit-learn.org/stable/modules/'
                             'generated/'
                             'sklearn.ensemble.RandomForestRegressor.html')))
                algorithms.append(
                    Algorithm(
                        Ridge(),
                        [{'alpha':
                            [0.01, 0.1, 1, 10, 100]}],
                        'Ridge Regression',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.linear_model.Ridge.html')))
                algorithms.append(
                    Algorithm(
                        DecisionTreeRegressor(random_state=self.random_state),
                        [{'max_depth': [3, 5, 7, 9, 11]}],
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeRegressor.html')))
            else:
                algorithms.append(
                    Algorithm(
                        SGDRegressor(
                            random_state=self.random_state),
                        [{'alpha': [1e-05, 3e-05, 1e-04, 3e-04, 1e-03]}],
                        'SGD Regressor',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.linear_model.SGDRegressor.html')))
        return algorithms

    def add_algorithm(self, estimator, param_grid, name, link=None):
        """
        Add arbitrary scikit-learn-compatible algorithm.

        Parameters
        ----------
        estimator : object type that implements the “fit” and “predict” methods
            A object of that type is instantiated for each grid point.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and
            lists of parameter settings to try as values, or a list of
            such dictionaries, in which case the grids spanned by
            each dictionary in the list are explored.
            This enables searching over any sequence of parameter settings.
        name : string
            Algorithm name (used for report)
        link : string
            URL to explain the algorithm (used for report)
        """
        if self.verbose:
            print('add %s' % name)
        self.algorithms.append(Algorithm(estimator, param_grid, name, link))

    def change_params(self, identifier, param_grid):
        """
        Change parameters of an algorithm.

        Parameters
        ----------
        identifier : integer or string.
            If an integer is passed, it is the index of the algorithm
            in the list of algorithms.
            If a string is passed, it is the name of the algorithm.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and
            lists of parameter settings to try as values, or a list of
            such dictionaries, in which case the grids spanned by
            each dictionary in the list are explored.
            This enables searching over any sequence of parameter settings.
        """
        if isinstance(identifier, int):
            self.algorithms[identifier].parameters = param_grid
        elif isinstance(identifier, str):
            for algorithm in self.algorithms:
                if algorithm.name == identifier:
                    algorithm.parameters = param_grid
                    break
        else:
            raise Exception('Wrong identifier')

    def remove_algorithm(self, index=-1):
        """
        Remove algorithm

        Parameters
        ----------
        index : int (default=-1)
            Remove an algorithm from list by index.
            By default, last algorithm is removed.
        """
        if self.verbose:
            print('remove %s' % self.algorithms[index].name)
        del self.algorithms[index]

    def get_algorithms(self):
        """
        Get algorithm names and grid parameters.

        Returns
        -------
        algorithms : list
            List of tupples(name, grid_params).
        """
        rtn = []
        for algorithm in self.algorithms:
            rtn.append((algorithm.name, algorithm.parameters))
        return rtn

    def fit(self, X, y, dname=None, algorithm_selection_only=False):
        """
        Tune parameters and search best algorithm

        Parameters
        ----------
        X : {numpy.ndarray, pandas.DataFrame}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : {numpy.ndarray, pandas.Series}, shape = [n_samples]
            Target values (class labels in classification, real numbers in
            regression)
        dname : string (default=None)
            If not None, make a analysis report in this directory.
        algorithm_selection_only : boolean, optional (default=False)
            If True, only algorithm selection is executed.
            This option is needed for (get|add|remove)_algorithm(s) methods.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.verbose:
            print('Set data.')
        self.data = Data(self.shuffle, self.standardize, self.random_state)
        self.data.fit_transform(X, y)

        if not self.is_ready:
            if self.verbose:
                print('Choose algorithm.')
            self.algorithms = self.__choose_algorithm()
            if self.verbose:
                for algorithm in self.algorithms:
                    print('    %s' % algorithm.name)
            self.is_ready = True
        else:
            # initialize
            for algorithm in self.algorithms:
                algorithm.best_score is None
                algorithm.best_params is None
                algorithm.is_best_algorithm = False
                algorithm.grid_scores is None
                algorithm.classification_report is None

        if algorithm_selection_only:
            return (self.data.X, self.data.y)

        if isinstance(self.cv, int):
            if self.task == 'classification':
                self.cv = StratifiedKFold(n_splits=self.cv,
                                          shuffle=self.shuffle,
                                          random_state=self.random_state)
            elif self.task == 'regression':
                self.cv = KFold(n_splits=self.cv,
                                shuffle=self.shuffle,
                                random_state=self.random_state)

        if self.verbose:
            print('Analyze. (take some time)')
        self.__tune_parameters()
        if self.task == 'classification':
            self.__report_classification_result()

        if dname is not None:
            if self.verbose:
                print('Make report.')
            self.__make_report(dname)

        self.results = {'algorithms': {}}
        for algorithm in self.algorithms:
            self.results['algorithms'][algorithm.name] = {}
            self.results['algorithms'][algorithm.name]['grid_scores'] =\
                algorithm.grid_scores

            if dname is None:
                self.results['algorithms'][algorithm.name]['learning_curve'] =\
                    self.__calc_learning_curve(algorithm)

            if algorithm.is_best_algorithm:
                self.results['best_algorithm'] = {}
                self.results['best_algorithm']['estimator'] =\
                    algorithm.estimator
                self.results['best_algorithm']['score'] = algorithm.best_score

        if self.verbose:
            print('Done.')
        return self

    def predict(self, X, estimator=None):
        if estimator is None:
            return self.algorithms[self.best_index].estimator.predict(
                self.data.transform(X))
        else:
            return estimator.predict(self.data.transform(X))

    def __search_best_algorithm(self):
        self.best_score = float('-Inf')
        self.best_index = -1
        for i in range(len(self.algorithms)):
            if self.algorithms[i].best_score > self.best_score:
                self.best_score = self.algorithms[i].best_score
                best_index = i
        self.algorithms[best_index].is_best_algorithm = True

    def __tune_parameters(self):
        for i in range(len(self.algorithms)):
            if self.verbose:
                print('    %s' % self.algorithms[i].name)
            estimator = self.algorithms[i].estimator
            parameters = self.algorithms[i].parameters
            clf = GridSearchCV(
                estimator, parameters, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs)
            clf.fit(self.data.X, self.data.y)
            grid_scores = []
            for j in range(len(clf.cv_results_['mean_test_score'])):
                grid_scores.append((clf.cv_results_['params'][j],
                                    clf.cv_results_['mean_test_score'][j],
                                    clf.cv_results_['std_test_score'][j]))
            self.algorithms[i].estimator = clf.best_estimator_
            self.algorithms[i].best_score = clf.best_score_
            self.algorithms[i].best_params = clf.best_params_
            self.algorithms[i].grid_scores = grid_scores

        self.__search_best_algorithm()

    def __report_classification_result(self):
        for i in range(len(self.algorithms)):
            est = self.algorithms[i].estimator
            self.algorithms[i].classification_report =\
                classification_report(self.data.y, est.predict(self.data.X))

    def __calc_learning_curve(self, algorithm):
        estimator = algorithm.estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            self.data.X,
            self.data.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs)  # parallel run in cross validation
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        return {'x': train_sizes, 'y_train': train_scores_mean,
                'y_cv': test_scores_mean}

    def __plot_learning_curve(self, dname=None):
        for alg in self.algorithms:
            if self.verbose:
                print('    %s' % alg.name)
            estimator = alg.estimator
            train_sizes, train_scores, test_scores = learning_curve(
                estimator,
                self.data.X,
                self.data.y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs)  # parallel run in cross validation
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure()
            plt.title(estimator.__class__.__name__)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std,
                             alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.legend(loc="lower right")
            if dname is not None and not os.path.exists(dname):
                os.mkdir(dname)
            if dname is not None:
                plt.savefig('%s/learning_curve_%s.png' %
                            (dname, estimator.__class__.__name__),
                            bbox_inches='tight', dpi=75)
            else:
                plt.savefig('learning_curve_%s.png' %
                            estimator.__class__.__name__,
                            bbox_inches='tight', dpi=75)
            plt.close()

    def __make_report(self, dname='report'):
        if not os.path.exists(dname):
            os.mkdir(dname)

        self.__plot_learning_curve(dname)

        env = Environment(
            loader=FileSystemLoader(
                os.path.abspath(
                    os.path.dirname(__file__)) + '/template', encoding='utf8'))
        if self.lang == 'jp':
            tmpl = env.get_template('report_jp.html.tmp')
        else:
            tmpl = env.get_template('report.html.tmp')

        scoring_name = self.scoring if isinstance(self.scoring, str) else\
            self.scoring.func_name
        html = tmpl.render(algorithms=self.algorithms,
                           scoring=scoring_name,
                           task=self.task,
                           data=self.data).encode('utf-8')
        fo = io.open(dname + '/report.html', 'w', encoding='utf-8')
        fo.write(html.decode('utf-8'))
        fo.close()

    def generate_module_sample(self, fname='module_sample.py'):
        """
        Generate a module sample to be able to add in the model
        in your system for prediction.

        Parameters
        ----------
        fname : string (default="module_sample.py")
            A string containing a path to a output file.
        """

        env = Environment(
            loader=FileSystemLoader(
                os.path.abspath(
                    os.path.dirname(__file__)) + '/template', encoding='utf8'))
        tmpl = env.get_template('sample_code.py.tmp')
        encoded = True if len(self.data.del_columns) > 0 else False
        html = tmpl.render(algorithm=self.algorithms[self.best_index],
                           encoded=encoded,
                           standardize=self.standardize).encode('utf-8')
        fo = io.open(fname, 'w', encoding='utf-8')
        fo.write(html.decode('utf-8'))
        fo.close()

    def select_features(self):
        if self.data is None:
            warnings.warn("'drop_col' must be used after 'fit' has used.")
            return

        if self.task == 'regression':
            rf = RandomForestRegressor(random_state=0, oob_score=True, n_estimators=50)
        else:
            rf = RandomForestClassifier(random_state=0, oob_score=True, n_estimators=50)
        
        num_col = len(self.data.X.columns)
        self.data.drop_col(rf)
        if len(self.data.X.columns) < num_col:
            self.algorithms = self.__choose_algorithm()
            self.is_ready = True


if __name__ == "__main__":
    MALSS(interactive=True)
