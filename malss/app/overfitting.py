# coding: utf-8

import os
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton,
                             QRadioButton, QButtonGroup)
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
            self.set_paragraph('Hyper-parameter tuning', text=text)
        else:
            self.set_paragraph('ハイパーパラメータチューニング', text=text, img=path1)

        hbox2 = QHBoxLayout()
        hbox2.setContentsMargins(10, 10, 10, 10)

        self.btn = QPushButton('Next', self.inner)
        self.btn.clicked.connect(lambda: self.button_func('Analysis'))

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn)

        self.vbox.addLayout(hbox2)

        self.vbox.addStretch(1)
