# coding: utf-8

import os
import pandas as pd
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

        self.vbox.addStretch(1)

        self.btn = QPushButton('Next', self.inner)
        self.btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        self.btn.clicked.connect(self.button_clicked)

        if self.params.fpath is not None:
            self.le.setText(self.params.fpath)
        else:
            self.btn.setEnabled(False)

        self.vbox.addWidget(self.btn)

    def show_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open', '')
        self.le.setText(fname[0])
        self.btn.setEnabled(True)
        # self.params.fpath = fname[0]
        self.fpath = fname[0]

    def button_clicked(self):
        if self.params.fpath != self.fpath:
            self.button_func('Error', delete=True)
            self.params.fpath = self.fpath
            if self.params.data5 is None:
                try:
                    # engine='python' is to avoid pandas's bug.
                    data = pd.read_csv(
                        self.params.fpath, header=0, engine='python',
                        dtype=self.make_dtype(self.params.columns,
                                            self.params.col_types))
                except Exception:
                    import traceback
                    self.params.error = traceback.format_exc()
                    self.button_func('Error')
                    return
                self.params.data5 = data.head(min(5, data.shape[0]))

                self.params.columns = data.columns
                self.params.col_types_def =\
                    list(map(str, data.dtypes.get_values()))
                self.params.col_types =\
                    list(map(str, data.dtypes.get_values()))

        if self.params.lang == 'jp':
            self.button_func('データの確認')
        else:
            self.button_func('Data check')
