# coding: utf-8

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton)
from .content import Content


class Introduction(Content):

    def __init__(self, parent=None, button_func=None):
        super().__init__(parent, 'Introduction')

        self.button_func = button_func

        text = \
"""abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz
abcdefg hijklmn opqrstu vwxyz abcdefg hijklmn opqrstu vwxyz"""

        self.set_paragraph('Introduction', text=text)

        # hbox = QHBoxLayout(self.inner)  # raise warning
        hbox = QHBoxLayout()
        hbox.setContentsMargins(10, 10, 10, 10)
        
        btn = QPushButton('hoge', self.inner)
        btn.clicked.connect(lambda: self.button_func('Hoge'))

        hbox.addStretch(1)
        hbox.addWidget(btn)

        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)
