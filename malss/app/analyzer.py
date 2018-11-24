# coding: utf-8

from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtCore import QThread, pyqtSignal
from .content import Content
from multiprocessing import Process, Queue
from .waiting_animation import WaitingAnimation


class Analyzer(Content):

    def __init__(self, parent=None, title='', params=None):
        super().__init__(parent, title, params)

        # "parent.parent()" must be modified.
        self.wait_ani = WaitingAnimation(parent.parent())
        self.wait_ani.hide()

    def resizeEvent(self, event):
        # To be modified.
        self.wait_ani.resize(self.parent().parent().size())
        event.accept()

        QScrollArea.resizeEvent(self, event)

    def preprocess(self):
        """
        This method need to be overridden.
        """
        pass

    def button_clicked(self, mdl, X, y, next_page):
        self.analyze(mdl, X, y, next_page)

    def analyze(self, mdl, X, y, next_page):
        if len(mdl.get_algorithms()) > 0:
            self.thread = AnalyzeWorker(mdl, X, y)
            self.thread.finSignal.connect(self.analyzed)
            self.thread.start()
            self.wait_ani.show()
        else:
            """
            Already analyzed and not need to re-analyze.
            """
            self.button_func(next_page)

    def analyzed(self, signalData):
        """
        This method need to be overridden.
        """
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            pass

    def add_algorithm(self, mdl, algorithms, results):
        """
        Add algorithm for re-analysis if hyper-parameters are changed.
        Otherwise, mdl.get_algorithms() returns empty list.
        """
        prev_algorithms = mdl.get_algorithms()
        for n in range(len(prev_algorithms)):
            mdl.remove_algorithm(0)

        for name, parameters in algorithms:
            if not self.__need_analyze(name, parameters, results):
                continue

            if name == 'Support Vector Machine (RBF Kernel)':
                if self.params.task == 'Regression':
                    from sklearn.svm import SVR
                    mdl.add_algorithm(
                        SVR(kernel='rbf'),
                        parameters,
                        'Support Vector Machine (RBF Kernel)',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/sklearn.svm.SVR.html'))
                elif self.params.task == 'Classification':
                    from sklearn.svm import SVC
                    mdl.add_algorithm(
                        SVC(random_state=mdl.random_state,
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
                    mdl.add_algorithm(
                        RandomForestRegressor(
                            random_state=mdl.random_state,
                            n_estimators=500,
                            n_jobs=1),
                        parameters,
                        'Random Forest',
                        ('http://scikit-learn.org/stable/modules/'
                         'generated/'
                         'sklearn.ensemble.RandomForestRegressor.html'))
                elif self.params.task == 'Classification':
                    from sklearn.ensemble import RandomForestClassifier
                    mdl.add_algorithm(
                        RandomForestClassifier(
                            random_state=mdl.random_state,
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
                mdl.add_algorithm(
                    LinearSVC(random_state=mdl.random_state),
                    parameters,
                    'Support Vector Machine (Linear Kernel)',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.svm.LinearSVC.html'))
            elif name == 'Logistic Regression':
                from sklearn.linear_model import LogisticRegression
                mdl.add_algorithm(
                    LogisticRegression(
                        random_state=mdl.random_state),
                    parameters,
                    'Logistic Regression',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.LogisticRegression.html'))
            elif name == 'Decision Tree':
                if self.params.task == 'Regression':
                    from sklearn.tree import DecisionTreeRegressor
                    mdl.add_algorithm(
                        DecisionTreeRegressor(
                            random_state=mdl.random_state),
                        parameters,
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeRegressor.html'))
                elif self.params.task == 'Classification':
                    from sklearn.tree import DecisionTreeClassifier
                    mdl.add_algorithm(
                        DecisionTreeClassifier(
                            random_state=mdl.random_state),
                        parameters,
                        'Decision Tree',
                        ('http://scikit-learn.org/stable/modules/generated/'
                         'sklearn.tree.DecisionTreeClassifier.html'))
                else:
                    raise Exception('Wrong task name.')
            elif name == 'k-Nearest Neighbors':
                from sklearn.neighbors import KNeighborsClassifier
                mdl.add_algorithm(
                    KNeighborsClassifier(),
                    parameters,
                    'k-Nearest Neighbors',
                    ('http://scikit-learn.org/stable/modules/'
                     'generated/sklearn.neighbors.KNeighborsClassifier'
                     '.html'))
            elif name == 'SGD Classifier':
                from sklearn.linear_model import SGDClassifier
                mdl.add_algorithm(
                    SGDClassifier(
                        random_state=mdl.random_state,
                        n_jobs=1),
                    parameters,
                    'SGD Classifier',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.SGDClassifier.html'))
            elif name == 'Ridge Regression':
                from sklearn.linear_model import Ridge
                mdl.add_algorithm(
                    Ridge(),
                    parameters,
                    'Ridge Regression',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.Ridge.html'))
            elif name == 'SGD Regressor':
                from sklearn.linear_model import SGDRegressor
                mdl.add_algorithm(
                    SGDRegressor(
                        random_state=mdl.random_state),
                    parameters,
                    'SGD Regressor',
                    ('http://scikit-learn.org/stable/modules/generated/'
                     'sklearn.linear_model.SGDRegressor.html'))

    def __need_analyze(self, name, parameters, results):
        flg = False

        param_result = results['algorithms'][name]['grid_scores']
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

        return flg


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
        rtn = q.get()
        job.join()
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
