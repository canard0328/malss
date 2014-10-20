# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
from sklearn import cross_validation, metrics
from sklearn.utils import shuffle as sk_shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import fetch_mldata
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from algorithm import Algorithm
from html import HTML


class MALSS(object):
    def __init__(self, X, y, task, shuffle=True, n_jobs=1, random_state=0,
                 verbose=True):
        """
        Set the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (class labels in classification, real numbers in
            regression)
        task : string
            Specifies the task of the analysis. It must be one of
            'classification', 'regression'.
        shuffle : boolean, optional (default=True)
            Whether to shuffle the data.
        n_jobs : integer, optional (default=1)
            The number of jobs to run in parallel. If -1, then the number of
            jobs is set to the number of cores.
        random_state : int seed, RandomState instance, or None (default=0)
            The seed of the pseudo random number generator
        verbos : bool, default: True
            Enable verbose output.
        """
        if task == 'classification':
            if shuffle:
                self.X, self.y = sk_shuffle(X, y, random_state=0)
            else:
                self.X = X
                self.y = y
            self.X = StandardScaler().fit_transform(self.X)
            self.task = task
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.verbose = verbose
            self.algorithms = self.__choose_algorithm()
            self.scoring = 'f1'
        elif task == 'regression':
            self.scoring = 'r2'
            raise ValueError('task:%s is not implemented yet' % task)
        else:
            raise ValueError('task:%s is not supported' % task)

    def __choose_algorithm(self):
        algorithms = []
        if self.task == 'classification':
            if self.X.shape[0] * self.X.shape[1] <= 1e+06:
                if self.X.shape[0] ** 2 * self.X.shape[1] <= 1e+09:
                    algorithms.append(
                        Algorithm(
                            SVC(random_state=self.random_state),
                            [{'kernel': ['rbf'],
                              'C': [10, 100, 1000, 10000],
                              'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}],
                            'Support Vector Machine (RBF Kernel)'))
                algorithms.append(
                    Algorithm(
                        LinearSVC(random_state=self.random_state),
                        [{'C': [0.1, 1, 10, 100]}],
                        'Support Vector Machine (Linear Kernel)'))
                algorithms.append(
                    Algorithm(
                        LogisticRegression(random_state=self.random_state),
                        [{'penalty': ['l2', 'l1'],
                          'C': [0.1, 0.3, 1, 3, 10],
                          'class_weight': [None, 'auto']}],
                        'Logistic Regression'))
                algorithms.append(
                    Algorithm(
                        DecisionTreeClassifier(random_state=self.random_state),
                        [{'max_depth': [3, 5, 7, 9, 11]}],
                        'Decision Tree'))
                algorithms.append(
                    Algorithm(
                        KNeighborsClassifier(),
                        [{'n_neighbors': [2, 6, 10, 14, 18]}],
                        'k-Nearest Neighbors'))
            else:
                algorithms.append(
                    Algorithm(
                        SGDClassifier(
                            random_state=self.random_state,
                            n_jobs=self.n_jobs),
                        [{'loss': ['hinge', 'log'],
                          'penalty': ['l2', 'l1'],
                          'alpha': [1e-05, 1e-04, 1e-03],
                          'class_weight': [None, 'auto']}],
                        'SGD Classifier'))
        return algorithms

    def execute(self):
        self.__tune_parameters()
        self.__report_classification_result()

    def __tune_parameters(self):
        for i in xrange(len(self.algorithms)):
            estimator = self.algorithms[i].estimator
            parameters = self.algorithms[i].parameters
            cv = cross_validation.StratifiedKFold(
                self.y, n_folds=5, shuffle=True,
                random_state=self.random_state)
            sc = f1score if self.scoring == 'f1' else self.scoring
            clf = GridSearchCV(
                estimator, parameters, cv=cv, scoring=sc,
                n_jobs=self.n_jobs)
            clf.fit(self.X, self.y)
            self.algorithms[i].estimator = clf.best_estimator_
            self.algorithms[i].best_score = clf.best_score_

            self.algorithms[i].description += HTML.h3('Parameter optimization')
            tbl, col = [], []
            for row, scr in enumerate(clf.grid_scores_):
                fc = '#FF0000' if scr[0] == clf.best_params_ else '#000000'
                if row == 0:
                    tbl.append(scr[0].keys() + ['%s' % self.scoring, 'SD'])
                    col.append('#000000')
                tbl.append(scr[0].values() +
                           ['%.3f' % scr[1], '%.3f' % scr[2].std()])
                col.append(fc)
            self.algorithms[i].description += HTML.table(tbl, col)

            if self.verbose:
                self.algorithms[i].description += \
                    HTML.list_item(['If the best parameter is at the border ' +
                                    'of the grid, its range should be ' +
                                    'expanded.\n',
                                    'Often a second, narrower grid is ' +
                                    'searched centered around the best ' +
                                    'parameters of the first grid.\n'])

    def __report_classification_result(self):
        for i in xrange(len(self.algorithms)):
            est = self.algorithms[i].estimator
            self.algorithms[i].description +=\
                HTML.h3('Classification report')
            self.algorithms[i].description\
                += HTML.pre(classification_report(self.y, est.predict(self.X)))

    def __plot_learning_curve(self, dname=None):
        for alg in self.algorithms:
            estimator = alg.estimator
            cv = cross_validation.StratifiedKFold(
                self.y, n_folds=5, shuffle=True,
                random_state=self.random_state)
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, self.X, self.y, cv=cv, n_jobs=self.n_jobs)
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

            alg.description += HTML.h3('Learning curve')
            alg.description +=\
                HTML.img('learning_curve_%s.png' %
                         alg.estimator.__class__.__name__,
                         300, 'learning_curve')

            if self.verbose:
                alg.description +=\
                    HTML.list_item_with_title([
                        ['High variance',
                         ['Cross-validation score still increasing as ' +
                          'training examples increases.',
                          'Large gap between training and cross-validation ' +
                          'score.']],
                        ['High bias',
                         ['Even training score is unacceptably low.',
                          'Small gap between training and cross-validation ' +
                          'score.']],
                        ['In case of high variance:',
                         ['Try getting more training examples.',
                          'Try dimensionality reduction or feature ' +
                          'selection.']],
                        ['In case of high bias:',
                         ['Try a larger set of features.']]])

    def make_report(self, dname='report'):
        """
        Make the report

        Parameters
        ----------
        dname : string (default="report")
            A string containing a path to a output directory.
        """

        if not os.path.exists(dname):
            os.mkdir(dname)

        self.__plot_learning_curve(dname)

        fo = HTML.open(dname + '/report.html', 'w')
        fo.write(HTML.h1('Results', 'top'))
        best_score = float('-Inf')
        for alg in self.algorithms:
            if alg.best_score > best_score:
                best_score = alg.best_score
        tbl, col, link = [], [], []
        tbl.append(['algorithm', 'score (%s)' % self.scoring])
        col.append('#000000')
        link.append(None)
        for alg in self.algorithms:
            fc = '#FF0000' if alg.best_score == best_score else '#000000'
            col.append(fc)
            tbl.append([alg.name, '%.5f' % alg.best_score])
            link.append('#%s' % alg.name)
        fo.write(HTML.table(tbl, col, link))
        fo.write(HTML.hr())
        for alg in self.algorithms:
            fo.write(alg.description)
            fo.write(HTML.hr())
        HTML.close(fo)


def f1score(estimator, X, y):
    return metrics.f1_score(y, estimator.predict(X), average=None).mean()


if __name__ == "__main__":
    pass
