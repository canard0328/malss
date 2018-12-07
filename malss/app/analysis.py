# coding: utf-8

import pandas as pd
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QTableWidgetItem)
from PyQt5.QtCore import Qt
from ..malss import MALSS
from .nonscroll_table import NonScrollTable
from .analyzer import Analyzer


class Analysis(Analyzer):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Analysis', params)

        self.button_func = button_func

        self.preprocess()

        if self.params.lang == 'en':
            text = ('First 5 rows of your features are shown below.\n'
                    'Confirm that the data was normalized and\n'
                    'categorical variables (if any) are converted into dummy '
                    'variables.\n'
                    '(Categorical variable A that has values X, Y, and Z are '
                    'converted into dummy variables, A_X, A_Y, and A_Z.)')
            self.set_paragraph('Features', text=text)
        else:
            text = ('説明変数の先頭5行を以下に示します．\n'
                    'データの正規化（Normalization）が行われていること，\n'
                    'カテゴリ変数があればダミー変数を用いて量的変数に変換されていることを確認してください．\n'
                    '（X, Y, Zという値をとるカテゴリ変数AはA_X, A_Y, A_Zという変数に変換されます）')
            self.set_paragraph('説明変数', text=text)

        nr = 5

        table1 = NonScrollTable(self.inner)

        table1.setRowCount(nr)
        table1.setColumnCount(len(self.params.X.columns))
        table1.setHorizontalHeaderLabels(self.params.X.columns)

        for r in range(nr):
            for c in range(len(self.params.X.columns)):
                item = QTableWidgetItem(str(round(self.params.X.iat[r, c], 4)))
                item.setFlags(Qt.ItemIsEnabled)
                table1.setItem(r, c, item)

        table1.setNonScroll()

        self.vbox.addWidget(table1)

        if self.params.lang == 'en':
            text = ('MALSS selected appropriate algorithms '
                    'according to your task and data.\n'
                    'Selected algorithms are shown below.')
            self.set_paragraph('Algorithms', text=text)
        else:
            text = ('MALSSは分析タスクとデータに応じて適切なアルゴリズムを選択します．\n'
                    'アルゴリズム選択結果を以下の示します．')
            self.set_paragraph('アルゴリズム', text=text)

        nr = len(self.params.algorithms)

        table2 = NonScrollTable(self.inner)

        table2.setRowCount(nr)
        table2.setColumnCount(1)
        table2.setHorizontalHeaderLabels(['Algorithms'])

        for r in range(nr):
            item = QTableWidgetItem(self.params.algorithms[r][0])
            item.setFlags(Qt.ItemIsEnabled)
            table2.setItem(r, 0, item)

        table2.setNonScroll()

        self.vbox.addWidget(table2)

        if self.params.lang == 'en':
            text = ('MALSS automatically perform cross validation analysis '
                    'with grid search hyper-parameter tuning.\n'
                    'Clik "Analyze" to start.\n'
                    '(It will take tens of minutes.)')
            self.set_paragraph('', text=text)
        else:
            text = ('MALSSはグリッドサーチによるハイパーパラメータチューニング，'
                    '交差検証を自動で行います．\n'
                    'それでは分析を始めましょう．Anayzeボタンを押してください．\n'
                    '（分析には数分～数十分かかります）')
            self.set_paragraph('', text=text)

        self.vbox.addStretch(1)

        btn = QPushButton('Analyze', self.inner)
        btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            next_page = 'Results'
        else:
            next_page = '結果の確認'
        btn.clicked.connect(lambda: self.button_clicked(
            self.params.mdl, self.params.X, self.params.y, next_page))

        self.vbox.addWidget(btn)

        lists = ['task', 'supervised_learning', 'dummy', 'hyperparameter',
                 'overfitting', 'cross_validation', 'learning_curve',
                 'bias_variance']
        if self.params.lang == 'jp':
            lists = [l + '_jp' for l in lists]
        else:
            lists = [l + '_en' for l in lists]
        self.wait_ani.set_lists(lists)

    def preprocess(self):
        if self.params.col_types_changed:
            try:
                # engine='python' is to avoid pandas's bug.
                data = pd.read_csv(self.params.fpath, header=0, engine='python',
                                dtype=self.make_dtype(self.params.columns,
                                                        self.params.col_types))
            except Exception:
                import traceback
                self.params.error = traceback.format_exc()
                self.button_func('Error')
                return

            self.params.col_types_changed = False

            X = data.drop(self.params.objective, axis=1)
            y = data.loc[:, self.params.objective]

            if self.params.mdl is None:
                self.params.mdl = MALSS(self.params.task.lower())
                self.params.X, self.params.y =\
                    self.params.mdl.fit(X, y,
                                        algorithm_selection_only=True)

                self.params.algorithms = self.params.mdl.get_algorithms()
        else:
            self.add_algorithm(self.params.mdl, self.params.algorithms,
                               self.params.results)

    def analyzed(self, signalData):
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            if self.params.results is None:
                self.params.results = signalData
            else:
                # Update results of re-analyzed algorithms
                for name, results in signalData['algorithms'].items():
                    self.params.results['algorithms'][name]['grid_scores'] =\
                        results['grid_scores']
                    self.params.results['algorithms'][name]['learning_curve'] =\
                        results['learning_curve']

                # Update best algorithm
                if signalData['best_algorithm']['score'] >\
                        self.params.results['best_algorithm']['score']:
                    self.params.results['best_algorithm']['estimator'] =\
                        signalData['best_algorithm']['estimator']
                    self.params.results['best_algorithm']['score'] =\
                        signalData['best_algorithm']['score']

            if self.params.lang == 'en':
                self.button_func('Results')
            else:
                self.button_func('結果の確認')
