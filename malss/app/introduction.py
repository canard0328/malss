# coding: utf-8

import os
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton)
from .content import Content


class Introduction(Content):

    def __init__(self, parent=None, button_func=None, lang='en'):
        super().__init__(parent, 'Introduction', lang)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'
        if self.lang == 'en':
            path += 'introduction_en.txt'
        else:
            path += 'introduction_jp.txt'

        text = self.get_text(path)

        self.set_paragraph('Introduction', text=text)

        # hbox = QHBoxLayout(self.inner)  # raise warning
        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)
        
        btn = QPushButton('Next', self.inner)
        btn.clicked.connect(lambda: self.button_func('Analysis'))

        hbox.addStretch(1)
        hbox.addWidget(btn)

        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)
