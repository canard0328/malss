# coding: utf-8

from PyQt5.QtWidgets import QPushButton
from .results_base import ResultsBase


class Results(ResultsBase):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Results', params)

        self.button_func = button_func

        self.make_tables(self.params.results['algorithms'],
                         self.params.algorithms)

        self.vbox.addStretch()

        btn_re = QPushButton('Re-analyze', self.inner)
        btn_re.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            btn_re.clicked.connect(lambda: self.button_func('Analysis'))
        else:
            btn_re.clicked.connect(lambda: self.button_func('分析の実行'))

        self.btn_next = QPushButton('Continue without any changes', self.inner)
        self.btn_next.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn_next.clicked.connect(lambda: self.button_func(
                'Bias and Variance'))
        else:
            self.btn_next.clicked.connect(lambda: self.button_func(
                'バイアスとバリアンス'))

        self.vbox.addWidget(btn_re)
        self.vbox.addWidget(self.btn_next)


class Results2(ResultsBase):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Results2', params)

        self.button_func = button_func

        self.make_tables(self.params.results_fs['algorithms'],
                         self.params.algorithms_fs)

        self.vbox.addStretch()

        btn_re = QPushButton('Re-analyze', self.inner)
        btn_re.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            btn_re.clicked.connect(
                lambda: self.button_func('Feature selection'))
        else:
            btn_re.clicked.connect(
                lambda: self.button_func('特徴量選択'))

        self.btn_next = QPushButton('Continue without any changes', self.inner)
        self.btn_next.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn_next.clicked.connect(lambda: self.button_func(
                'Learning curve 2'))
        else:
            self.btn_next.clicked.connect(lambda: self.button_func(
                '学習曲線２'))

        self.vbox.addWidget(btn_re)
        self.vbox.addWidget(self.btn_next)
