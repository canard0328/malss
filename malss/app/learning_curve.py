# coding: utf-8

import os
import numpy as np
from PyQt5.QtWidgets import (QPushButton, QScrollArea, QSizePolicy)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr
from multiprocessing import Process, Queue
from .content import Content
from .waiting_animation import WaitingAnimation
from .rfpimp import oob_importances


class LearningCurve(Content):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'LearningCurve', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        path1 = path + 'check_curve'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('', text=text)
        else:
            self.set_paragraph('', text=text)

        ylim = self.__get_ylim(self.params.results['algorithms'])
        for name, val in self.params.results['algorithms'].items():
            self.set_paragraph(h2=name)

            x = val['learning_curve']['x']
            y_train = val['learning_curve']['y_train']
            y_cv = val['learning_curve']['y_cv']
            fig = PlotLearningCurve(x, y_train, y_cv, ylim, name, self.inner)
            self.vbox.addWidget(fig)

        self.vbox.addStretch()

        btn_fs = QPushButton('Try feature selection', self.inner)
        btn_fs.clicked.connect(self.__button_clicked)

        self.btn_next = QPushButton('Continue', self.inner)
        if self.params.lang == 'en':
            self.btn_next.clicked.connect(lambda: self.button_func(
                'Prediction'))
        else:
            self.btn_next.clicked.connect(lambda: self.button_func(
                '予測'))

        self.vbox.addWidget(btn_fs)
        self.vbox.addWidget(self.btn_next)

        # "parent.parent()" must be modified.
        self.wait_ani = WaitingAnimation(parent.parent())
        self.wait_ani.hide()

    def __get_ylim(self, algorithms):
        ymin = float('Inf')
        ymax = -float('Inf')

        for name, val in algorithms.items():
            ymin = min([ymin, min(val['learning_curve']['y_cv'])])
            ymax = max([ymax, max(val['learning_curve']['y_train'])])
        margin = 0.05 * (ymax - ymin)

        return (ymin - margin, ymax + margin)

    def resizeEvent(self, event):
        # To be modified.
        self.wait_ani.resize(self.parent().parent().size())
        event.accept()

        QScrollArea.resizeEvent(self, event)

    def __button_clicked(self):
        self.__feature_selection()

    def __feature_selection(self):
        if self.params.task.lower() == 'regression':
            rf = RFr(random_state=0, oob_score=True, n_estimators=50)
        else:
            rf = RFc(random_state=0, oob_score=True, n_estimators=50)

        self.thread = FeatureSelectionWorker(rf, self.params.X.copy(deep=True),
                                             self.params.y)
        self.thread.finSignal.connect(self.__feature_selected)
        self.thread.start()
        self.wait_ani.show()

    def __feature_selected(self, signalData):
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            if len(signalData['col']) < len(self.params.X.columns):
                # some features deleted
                self.params.X_fs = self.params.X[signalData['col']]

                if self.params.lang == 'en':
                    self.button_func('Feature selection')
                else:
                    self.button_func('特徴量選択')
            else:
                # no features deleted
                self.params.not_deleted = True
                if self.params.lang == 'en':
                    self.button_func('Prediction')
                else:
                    self.button_func('予測')


class FeatureSelectionWorker(QThread):
    finSignal = pyqtSignal(dict)

    def __init__(self, mdl, X, y):
        super().__init__()
        self.mdl = mdl
        self.X = X
        self.y = y

    def run(self):
        q = Queue()
        job = Process(target=FeatureSelectionWorker.sub_job,
                      args=(self.mdl, self.X, self.y, q))
        job.start()
        job.join()
        rtn = q.get()
        self.finSignal.emit(rtn)

    @staticmethod
    def sub_job(mdl, X, y, q):
        rtn = {}
        try:
            while True:
                col, X = FeatureSelectionWorker.drop_col(X, y, mdl)
                if col is None:
                    break
            rtn['col'] = list(X.columns)
        except Exception as e:
            import traceback
            rtn['error'] = traceback.format_exc()
        q.put(rtn)

    @staticmethod
    def drop_col(X, y, rf, k=10, thr=0.0):
        col = None
        rf.fit(X, y)
        np.random.seed(0)
        imp = oob_importances(rf, X, y, n_samples=len(X))
        for i in range(1, k):
            imp += oob_importances(rf, X, y, n_samples=len(X))
        imp /= k
        if imp['Importance'].min() < thr:
            col = imp['Importance'].idxmin()
            print('Drop {}'.format(col))
            X = X.drop(col, axis=1)
        return col, X


class PlotLearningCurve(FigureCanvas):
    def __init__(self, x, y_train, y_cv, ylim, title, parent=None,
                 width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed, QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.plot(x, y_train, y_cv, ylim, title)

    def plot(self, x, y_train, y_cv, ylim, title):
        ax = self.figure.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.grid(True)

        ax.plot(x, y_train, 'o-', color='dodgerblue', label='Training score')
        ax.plot(x, y_cv, 'o-', color='darkorange',
                label='Cross-validation score')
        ax.set_ylim(ylim)
        ax.legend(loc="lower right")

        self.draw()
