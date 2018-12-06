# coding: utf-8

import os
from PyQt5.QtWidgets import QHBoxLayout, QPushButton
from .content import Content


class Overfitting(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Overfitting', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        # Text for hyper-parameter tuning
        path1 = path + 'hyperparameter'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('Hyper-parameter tuning', text=text, img=path1)
        else:
            self.set_paragraph('ハイパーパラメータチューニング', text=text, img=path1)

        # Text for overfitting
        path2 = path + 'overfitting'
        text = self.get_text(path2)
        if self.params.lang == 'en':
            self.set_paragraph('Overfitting', text=text, img=path2)
        else:
            self.set_paragraph('過学習', text=text, img=path2)

        # Text for cross validation
        path3 = path + 'cross_validation'
        text = self.get_text(path3)
        if self.params.lang == 'en':
            self.set_paragraph('Cross validation', text=text, img=path3)
        else:
            self.set_paragraph('交差検証', text=text, img=path3)

        self.vbox.addStretch(1)

        self.btn = QPushButton('Next', self.inner)
        self.btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn.clicked.connect(lambda: self.button_func('Analysis'))
        else:
            self.btn.clicked.connect(lambda: self.button_func('分析の実行'))

        self.vbox.addWidget(self.btn)
