# coding: utf-8

import os
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton)
from .content import Content


class Introduction(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Introduction', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'
        path += 'introduction'

        text = self.get_text(path)

        self.set_paragraph('MALSS interactive', text=text)

        # hbox = QHBoxLayout(self.inner)  # raise warning
        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)

        btn = QPushButton('Next', self.inner)

        if self.params.lang == 'en':
            btn.clicked.connect(lambda: self.button_func('Task'))
        else:
            btn.clicked.connect(lambda: self.button_func('分析タスク'))

        hbox.addStretch(1)
        hbox.addWidget(btn)

        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)
