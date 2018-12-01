# coding: utf-8

import os
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QLineEdit)
from .content import Content


class SetFile(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Input data', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        # Text for data format
        path1 = path + 'format'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('Data format', text=text, img=path1)
        else:
            self.set_paragraph('入力データフォーマット', text=text, img=path1)

        # Text for data format
        path2 = path + 'dummy'
        text = self.get_text(path2)
        if self.params.lang == 'en':
            self.set_paragraph('Dummy variables', text=text, img=path2)
        else:
            self.set_paragraph('ダミー変数', text=text, img=path2)

        if params.lang == 'jp':
            self.set_paragraph(
                'ファイル選択',
                text='データ分析を行うファイルを選択してください。')
        else:
            self.set_paragraph(
                'File selection',
                text='Choose your input file.')

        hbox1 = QHBoxLayout()
        hbox1.setContentsMargins(10, 10, 10, 10)

        lbl = QLabel('Location:', self.inner)
        self.le = QLineEdit(self.inner)
        fo_btn = QPushButton('Browse...', self.inner)
        fo_btn.clicked.connect(self.show_dialog)

        hbox1.addWidget(lbl)
        hbox1.addWidget(self.le)
        hbox1.addWidget(fo_btn)

        self.vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        hbox2.setContentsMargins(10, 10, 10, 10)

        self.btn = QPushButton('Next', self.inner)
        if params.lang == 'jp':
            self.btn.clicked.connect(lambda: self.button_func('データの確認'))
        else:
            self.btn.clicked.connect(lambda: self.button_func('Data check'))

        if self.params.fpath is not None:
            self.le.setText(self.params.fpath)
        else:
            self.btn.setEnabled(False)

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn)

        self.vbox.addLayout(hbox2)

        self.vbox.addStretch(1)

    def show_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open', '')
        self.le.setText(fname[0])
        self.btn.setEnabled(True)
        self.params.fpath = fname[0]
