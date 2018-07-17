# coding: utf-8

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton)
from ..malss import MALSS
from .content import Content

class Analysis(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Analysis')

        self.button_func = button_func

        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)
        
        btn = QPushButton('Analyze', self.inner)
        btn.clicked.connect(self.button_clicked)

        hbox.addStretch(1)
        hbox.addWidget(btn)

        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)

    def button_clicked(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        clf = MALSS('classification')
        clf.fit(iris.data, iris.target)
        print('done')

        self.button_func('Hoge')
