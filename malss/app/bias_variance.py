# coding: utf-8

import os
from PyQt5.QtWidgets import QHBoxLayout, QPushButton
from .content import Content


class BiasVariance(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Bias and Variance', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        # Text for learning curve
        path1 = path + 'learning_curve'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('Learning curve', text=text, img=path1)
        else:
            self.set_paragraph('学習曲線', text=text, img=path1)

        # Text for bias and variance
        path2 = path + 'bias_variance'
        text = self.get_text(path2)
        if self.params.lang == 'en':
            self.set_paragraph('Bias and Variance', text=text, img=path2)
        else:
            self.set_paragraph('バイアスとバリアンス', text=text, img=path2)

        if self.params.lang == 'en':
            text = ("Let's check out the learning curves. "
                    'Click "Next" to continue.')
            self.set_paragraph('', text=text)
        else:
            text = ('それでは学習曲線を確認してみましょう．'
                    'Nextボタンを押してください．')
            self.set_paragraph('', text=text)

        self.vbox.addStretch(1)

        self.btn = QPushButton('Next', self.inner)
        self.btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn.clicked.connect(lambda: self.button_func(
                'Learning curve'))
        else:
            self.btn.clicked.connect(lambda: self.button_func('学習曲線'))

        self.vbox.addWidget(self.btn)
