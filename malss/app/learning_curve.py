# coding: utf-8

import os
import numpy as np
import copy
from PyQt5.QtWidgets import (QPushButton, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr
from multiprocessing import Process, Queue
from ..malss import MALSS
from .waiting_animation import WaitingAnimation
from .rfpimp import oob_importances
from .learning_curve_base import LearningCurveBase


class LearningCurve(LearningCurveBase):
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

        self.plot_curve(self.params.results['algorithms'])

        self.vbox.addStretch()

        btn_fs = QPushButton('Try feature selection', self.inner)
        btn_fs.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        btn_fs.clicked.connect(self.__button_clicked)

        self.btn_next = QPushButton('Continue', self.inner)
        self.btn_next.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
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

        lists = ['task', 'supervised_learning', 'dummy', 'hyperparameter',
                 'overfitting', 'cross_validation', 'learning_curve',
                 'bias_variance']
        if self.params.lang == 'jp':
            lists = [l + '_jp' for l in lists]
        else:
            lists = [l + '_en' for l in lists]
        self.wait_ani.set_lists(lists)

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

        self.mdl_fs = copy.deepcopy(self.params.mdl)

        self.thread = FeatureSelectionWorker(self.mdl_fs)
        self.thread.finSignal.connect(self.__feature_selected)
        self.thread.start()
        self.wait_ani.show()

    def __feature_selected(self, signalData):
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            if len(signalData['mdl'].data.X.columns) < len(self.params.X.columns):
                # some features deleted
                self.params.X_fs = signalData['mdl'].data.X

                self.params.mdl_fs = signalData['mdl']

                self.params.algorithms_fs = self.params.mdl_fs.get_algorithms()
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


class LearningCurve2(LearningCurveBase):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'LearningCurve 2', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        path1 = path + 'learning_curve_2'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('', text=text)
        else:
            self.set_paragraph('', text=text)

        self.plot_curve(self.params.results_fs['algorithms'])

        if self.params.lang == 'en':
            text = ('Finally, MALSS output analysis results, and you can '
                    'predict unknown data (if you have).\n'
                    'Press "Next" to continue.')
            self.set_paragraph('', text=text)
        else:
            text = ('最後に学習結果の出力と，未知データがあればその予測を'
                    '行いましょう．\nNextを押してください')
            self.set_paragraph('', text=text)

        self.vbox.addStretch()

        self.btn_next = QPushButton('Next', self.inner)
        self.btn_next.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn_next.clicked.connect(lambda: self.button_func(
                'Prediction'))
        else:
            self.btn_next.clicked.connect(lambda: self.button_func(
                '予測'))

        self.vbox.addWidget(self.btn_next)


class FeatureSelectionWorker(QThread):
    finSignal = pyqtSignal(dict)

    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

    def run(self):
        q = Queue()
        job = Process(target=FeatureSelectionWorker.sub_job,
                      args=(self.mdl, q))
        job.start()
        rtn = q.get()
        job.join()
        self.finSignal.emit(rtn)

    @staticmethod
    def sub_job(mdl, q):
        rtn = {}
        try:
            mdl.select_features()
            rtn['mdl'] = mdl
        except Exception as e:
            import traceback
            rtn['error'] = traceback.format_exc()
        q.put(rtn)