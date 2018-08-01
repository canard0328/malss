# coding: utf-8

import pandas as pd
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton,
        QScrollArea)
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

    def resizeEvent(self, event):
        # To be modified.
        self.wait_ani.resize(self.parent().parent().size())
        event.accept()

        QScrollArea.resizeEvent(self, event)

    def button_clicked(self):
        self.params.data = pd.read_csv(self.params.fpath, header=0,
                dtype=self.make_dtype(self.params.columns, self.params.col_types))
        col_cat = [self.params.columns[i] for i in range(len(self.params.columns))\
                if self.params.col_types[i] == 'object']
        data_tidy = pd.get_dummies(self.params.data, columns=col_cat, drop_first=True)
        X, y = data_tidy.drop(self.params.objective, axis=1), data_tidy.loc[:, self.params.objective]
        
        self.thread = AnalyzeWorker(self.params.task.lower(), X, y)
        self.thread.finSignal.connect(self.analyzed)
        self.thread.start()
        self.wait_ani.show()

    def analyzed(self, signalData):
        self.wait_ani.hide()
        print(signalData)
        self.button_func('Hoge')


class AnalyzeWorker(QThread):
    finSignal = pyqtSignal(dict)
    
    def __init__(self, task, X, y):
        super().__init__()
        self.mdl = MALSS(task)
        self.X = X
        self.y = y

    def run(self):
        self.mdl.fit(self.X, self.y, algorithm_selection_only=True)
        self.mdl.remove_algorithm(0)
        self.mdl.remove_algorithm(0)

        q = Queue()
        job = Process(target=AnalyzeWorker.sub_job, args=(self.mdl, self.X, self.y, q))
        job.start()
        job.join()
        rtn = q.get()
        self.finSignal.emit(rtn)

    @staticmethod
    def sub_job(mdl, X, y, q):
        mdl.fit(X, y)
        q.put(mdl.results)
