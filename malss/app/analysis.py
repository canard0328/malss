# coding: utf-8

import pandas as pd
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal
from ..malss import MALSS
from .content import Content
from multiprocessing import Process, Queue
from .waiting_animation import WaitingAnimation


class Analysis(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Analysis', params)

        self.button_func = button_func

        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)

        btn = QPushButton('Analyze', self.inner)
        btn.clicked.connect(self.button_clicked)

        hbox.addStretch(1)
        hbox.addWidget(btn)

        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)

        # To be modified.
        self.wait_ani = WaitingAnimation(parent.parent())
        self.wait_ani.hide()

        if self.params.algorithms is not None:
            self.analyze()

    def resizeEvent(self, event):
        # To be modified.
        self.wait_ani.resize(self.parent().parent().size())
        event.accept()

        QScrollArea.resizeEvent(self, event)

    def button_clicked(self):
        self.analyze()

    def analyze(self):
        self.params.data =\
                pd.read_csv(self.params.fpath, header=0,
                            dtype=self.make_dtype(self.params.columns,
                                                  self.params.col_types))
        col_cat = [self.params.columns[i]
                   for i in range(len(self.params.columns))
                   if self.params.col_types[i] == 'object']
        data_tidy = pd.get_dummies(self.params.data, columns=col_cat,
                                   drop_first=True)
        self.params.X = data_tidy.drop(self.params.objective, axis=1)
        self.params.y = data_tidy.loc[:, self.params.objective]

        if self.params.mdl is None:
            self.params.mdl = MALSS(self.params.task.lower())
            self.params.mdl.fit(self.params.X, self.params.y,
                                algorithm_selection_only=True)
            # self.params.mdl.remove_algorithm(-1)
            # self.params.mdl.remove_algorithm(-1)
            # self.params.mdl.remove_algorithm(-1)
            # self.params.mdl.remove_algorithm(0)
            # self.params.mdl.remove_algorithm(0)
            # self.params.mdl.remove_algorithm(0)
            self.params.algorithms = self.params.mdl.get_algorithms()
        else:
            self.__add_algorithm()

        self.thread = AnalyzeWorker(self.params.mdl, self.params.X,
                                    self.params.y)
        self.thread.finSignal.connect(self.analyzed)
        self.thread.start()
        self.wait_ani.show()

    def analyzed(self, signalData):
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            if self.params.results is None:
                self.params.results = signalData
            else:
                for name, results in signalData['algorithms'].items():
                    self.params.results['algorithms'][name]['grid_scores'] =\
                        results['grid_scores']
            self.button_func('Results')

    def __need_analyze(self, name, parameters):
        flg = False

        param_result = self.params.results['algorithms'][name]['grid_scores']
        param_dic = {}
        for param, score, std in param_result:
            for k, v in param.items():
                if k not in param_dic:
                    param_dic[k] = [v]
                else:
                    param_dic[k].append(v)
        for k in param.keys():
            param_dic[k] = sorted(list(set(param_dic[k])))

        for k, v in parameters[0].items():
            if param_dic[k][0] != v[0] or param_dic[k][-1] != v[-1]:
                flg = True
                break
            elif len(param_dic[k]) != len(v):
                flg = True
                break

        print('{},{},{},{}'.format(flg, param_dic, name, parameters))
        return flg

    def __add_algorithm(self):
        """
        Add algorithm for re-analysis if hyper-parameters are changed.
        """
        prev_algorithms = self.params.mdl.get_algorithms()
        for n in range(len(prev_algorithms)):
            self.params.mdl.remove_algorithm(0)

        for name, parameters in self.params.algorithms:
            if not self.__need_analyze(name, parameters):
                continue

            if name == 'Support Vector Machine (RBF Kernel)':
                if self.params.task == 'Regression':
                    from sklearn.svm import SVR
                    self.params.mdl.add_algorithm(
                        SVR(kernel='rbf'),
                        parameters,
                        'Support Vector Machine (RBF Kernel)',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/sklearn.svm.SVR.html'))
                elif self.params.task == 'Classification':
                    from sklearn.svm import SVC
                    self.params.mdl.add_algorithm(
                        SVC(random_state=self.params.mdl.random_state,
                            kernel='rbf'),
                        parameters,
                        'Support Vector Machine (RBF Kernel)',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/sklearn.svm.SVC.html'))
                else:
                    raise Exception('Wrong task name.')
            elif name == 'Random Forest':
                if self.params.task == 'Regression':
                    from sklearn.ensemble import RandomForestRegressor
                    self.params.mdl.add_algorithm(
                        RandomForestRegressor(
                            random_state=self.params.mdl.random_state,
                            n_estimators=500,
                            n_jobs=1),
                        parameters,
                        'Random Forest',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/'
                         'sklearn.ensemble.RandomForestRegressor.html'))
                elif self.params.task == 'Classification':
                    from sklearn.ensemble import RandomForestClassifier
                    self.params.mdl.add_algorithm(
                        RandomForestClassifier(
                            random_state=self.params.mdl.random_state,
                            n_estimators=500,
                            n_jobs=1),
                        parameters,
                        'Random Forest',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/'
                         'sklearn.ensemble.RandomForestClassifier.html'))
                else:
                    raise Exception('Wrong task name.')
            elif name == 'Support Vector Machine (Linear Kernel)':
                from sklearn.svm import LinearSVC
                self.params.mdl.add_algorithm(
                    LinearSVC(random_state=self.params.mdl.random_state),
                    parameters,
                    'Support Vector Machine (Linear Kernel)',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.svm.LinearSVC.html'))
            elif name == 'Logistic Regression':
                from sklearn.linear_model import LogisticRegression
                self.params.mdl.add_algorithm(
                    LogisticRegression(
                        random_state=self.params.mdl.random_state),
                    parameters,
                    'Logistic Regression',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.LogisticRegression.html'))
            elif name == 'Decision Tree':
                if self.params.task == 'Regression':
                    from sklearn.tree import DecisionTreeRegressor
                    self.params.mdl.add_algorithm(
                        DecisionTreeRegressor(
                            random_state=self.params.mdl.random_state),
                        parameters,
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeRegressor.html'))
                elif self.params.task == 'Classification':
                    from sklearn.tree import DecisionTreeClassifier
                    self.params.mdl.add_algorithm(
                        DecisionTreeClassifier(
                            random_state=self.params.mdl.random_state),
                        parameters,
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeClassifier.html'))
                else:
                    raise Exception('Wrong task name.')
            elif name == 'k-Nearest Neighbors':
                from sklearn.neighbors import KNeighborsClassifier
                self.params.mdl.add_algorithm(
                    KNeighborsClassifier(),
                    parameters,
                    'k-Nearest Neighbors',
                    ('http://scikit-learn.org/stable/modules/'
                     'generated/sklearn.neighbors.KNeighborsClassifier'
                     '.html'))
            elif name == 'SGD Classifier':
                from sklearn.linear_model import SGDClassifier
                self.params.mdl.add_algorithm(
                    SGDClassifier(
                        random_state=self.params.mdl.random_state,
                        n_jobs=1),
                    parameters,
                    'SGD Classifier',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.SGDClassifier.html'))
            elif name == 'Ridge Regression':
                from sklearn.linear_model import Ridge
                self.params.mdl.add_algorithm(
                    Ridge(),
                    parameters,
                    'Ridge Regression',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.Ridge.html'))
            elif name == 'SGD Regressor':
                from sklearn.linear_model import SGDRegressor
                self.params.mdl.add_algorithm(
                    SGDRegressor(
                        random_state=self.params.mdl.random_state),
                    parameters,
                    'SGD Regressor',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.SGDRegressor.html'))


class AnalyzeWorker(QThread):
    finSignal = pyqtSignal(dict)

    def __init__(self, mdl, X, y):
        super().__init__()
        self.mdl = mdl
        self.X = X
        self.y = y

    def run(self):
        q = Queue()
        job = Process(target=AnalyzeWorker.sub_job,
                      args=(self.mdl, self.X, self.y, q))
        job.start()
        job.join()
        rtn = q.get()
        self.finSignal.emit(rtn)

    @staticmethod
    def sub_job(mdl, X, y, q):
        rtn = {}
        try:
            mdl.fit(X, y)
            rtn = mdl.results
        except Exception as e:
            import traceback
            rtn['error'] = traceback.format_exc()
        q.put(rtn)
