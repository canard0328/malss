# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import types
from sklearn.utils import shuffle as sk_shuffle
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report

from algorithm import Algorithm


class MALSS(object):
    def __init__(self, X, y, task, shuffle=True, n_jobs=1, verbose=True):
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
            algorithms.append(Algorithm(SVC(random_state=0),
                                        [{'kernel': ['rbf'],
                                          'C': [10, 100, 1000, 10000],
                                          'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}],
                                        'Support Vector Machine'))
        return algorithms

    def execute(self):
        self.__tune_parameters()
        self.__report_classification_result()
        self.__plot_learning_curve()
        self.__make_report()

    def __tune_parameters(self):
        for i in xrange(len(self.algorithms)):
            estimator = self.algorithms[i].estimator
            parameters = self.algorithms[i].parameters
            cv = cross_validation.StratifiedKFold(
                self.y, n_folds=5, shuffle=True, random_state=0)
            sc = f1score if self.scoring == 'f1' else self.scoring
            clf = GridSearchCV(
                estimator, parameters, cv=cv, scoring=sc,
                n_jobs=self.n_jobs)
            clf.fit(self.X, self.y)
            self.algorithms[i].estimator = clf.best_estimator_
            self.algorithms[i].best_score = clf.best_score_

            self.algorithms[i].description\
                += '<h3>Parameter optimization</h3>\n'
            self.algorithms[i].description\
                += '<table border="1" cellspacing="0" cellpadding="5">\n'
            for row, gs in enumerate(clf.grid_scores_):
                fc = '#FF0000' if gs[0] == clf.best_params_ else '#000000'
                self.algorithms[i].description += '<tr>\n'
                if row == 0:
                    for k in gs[0].keys():
                        self.algorithms[i].description += '<th>%s</th>\n' % k
                    self.algorithms[i].description += '<th>%s</th>\n' %\
                        self.scoring
                    self.algorithms[i].description += '<th>SD</th>\n'
                    self.algorithms[i].description += '</tr>\n'
                    self.algorithms[i].description += '<tr>\n'
                for v in gs[0].values():
                    self.algorithms[i].description\
                        += '<td><font color=%s>%s</font></td>\n' %\
                        (fc, v)
                self.algorithms[i].description\
                    += '<td><font color=%s>%.3f</font></td>\n' %\
                    (fc, gs[1])
                self.algorithms[i].description\
                    += '<td><font color=%s>%.3f</font></td>\n' %\
                    (fc, gs[2].std())
                self.algorithms[i].description += '</tr>\n'
            self.algorithms[i].description += '</table>\n\n'
            
            if self.verbose:
                self.algorithms[i].description += '<p>\n'
                self.algorithms[i].description += '<ul>\n'
                self.algorithms[i].description += '<li>If the best parameter is at the border of the grid, its range should be expanded.\n'
                self.algorithms[i].description += '<li>Often a second, narrower grid is searched centered around the best parameters of the first grid.\n'
                self.algorithms[i].description += '</ul>\n'
                self.algorithms[i].description += '</p>\n\n'

    def __report_classification_result(self):
        for i in xrange(len(self.algorithms)):
            est = self.algorithms[i].estimator
            self.algorithms[i].description +=\
                '<h3>Classification report </h3>\n'
            self.algorithms[i].description += '<pre>\n'
            self.algorithms[i].description +=\
                classification_report(self.y,
                                      est.predict(self.X))
            self.algorithms[i].description += '</pre>\n\n'

    def __plot_learning_curve(self):
        for alg in self.algorithms:
            estimator = alg.estimator
            cv = cross_validation.StratifiedKFold(
                self.y, n_folds=5, shuffle=True, random_state=0)
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
            plt.savefig('learning_curve_%s.png' % estimator.__class__.__name__,
                        bbox_inches='tight', dpi=75)
            plt.close()

            alg.description += '<h3>Learning curve</h3>\n'
            alg.description += '<img border="0" src="learning_curve_%s.png" height="300" alt="learning_curve">\n' %\
                alg.estimator.__class__.__name__
            alg.description += '\n'

            if self.verbose:
                alg.description += '<p>\n'
                alg.description += '<strong>High variance</strong>\n'
                alg.description += '<ul>\n'
                alg.description += '<li>Cross-validation score still increasing as training examples increases.</li>\n'
                alg.description += '<li>Large gap between training and cross-validation score.</li>\n'
                alg.description += '</ul>\n'
                alg.description += '<strong>High bias</strong>\n'
                alg.description += '<ul>\n'
                alg.description += '<li>Even training score is unacceptably low.</li>\n'
                alg.description += '<li>Small gap between training and cross-validation score.</li>\n'
                alg.description += '</ul>\n'
                alg.description += '<strong>In case of high variance:</strong>\n'
                alg.description += '<ul>\n'
                alg.description += '<li>Try getting more training examples.</li>\n'
                alg.description += '<li>Try dimensionality reduction or feature selection.</li>\n'
                alg.description += '</ul>\n'
                alg.description += '<strong>In case of high bias:</strong>\n'
                alg.description += '<ul>\n'
                alg.description += '<li>Try a larger set of features.</li>\n'
                alg.description += '</ul>\n'
                alg.description += '</p>\n\n'

    def __make_report(self):
        fo = open('report.html', 'w')
        fo.write('<html>\n\n')
        fo.write('<head>\n')
        fo.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n')
        fo.write('<title>Analysis report</title>')
        fo.write('</head>\n\n')
        fo.write('<body>\n')
        fo.write('<h1>Results</h1>\n')
        best_score = float('-Inf')
        for alg in self.algorithms:
            if alg.best_score > best_score:
                best_score = alg.best_score
        fo.write('<table border="1" cellspacing="0" cellpadding="5">\n')
        fo.write('<tr>\n<th>algorithm</th>\n<th>score (%s)</th>\n</tr>\n' % self.scoring)
        for alg in self.algorithms:
            fc = '#FF0000' if alg.best_score == best_score else '#000000'
            fo.write('<tr>\n')
            fo.write('<td><font color=%s>%s</font></td>\n' % (fc, alg.name))
            fo.write('<td><font color=%s>%s</font></td>\n' % (fc, alg.best_score))
            fo.write('</tr>\n')
        fo.write('</table>\n<hr>\n\n')
        for alg in self.algorithms:
            fo.write(alg.description)
        fo.write('</body>\n</html>\n')
        fo.close()


def f1score(estimator, X, y):
    return metrics.f1_score(y, estimator.predict(X), average=None).mean()


if __name__ == "__main__":
    pass
