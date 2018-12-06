# coding: utf-8

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton)
from PyQt5.QtCore import QCoreApplication
from .content import Content


class Error(Content):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Error', params)

        if self.params.lang == 'en':
            text = ('Unexpected error occured.\n'
                    'Please exit the application and solve the problem.')
        else:
            text = ('予期せぬエラーが発生しました。\n'
                    'アプリケーションを終了し，問題を解決してください。')
        self.set_paragraph('Unexpected error', text=text)

        self.set_paragraph('Traceback log', text=self.params.error)

        self.vbox.addStretch(1)

        btn = QPushButton('Exit', self.inner)
        btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        btn.clicked.connect(QCoreApplication.instance().quit)

        self.vbox.addWidget(btn)
