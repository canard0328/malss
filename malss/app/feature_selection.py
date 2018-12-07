# coding: utf-8

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QTableWidgetItem)
from PyQt5.QtCore import Qt
from .nonscroll_table import NonScrollTable
from .analyzer import Analyzer


class FeatureSelection(Analyzer):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Feature selection', params)

        self.button_func = button_func

        self.preprocess()

        if self.params.lang == 'en':
            text = ('Selected features are shown below.')
            self.set_paragraph('Features', text=text)
        else:
            text = ('特徴量選択の結果を以下に示します．')
            self.set_paragraph('説明変数', text=text)

        nr = len(self.params.X.columns)

        table1 = NonScrollTable(self.inner)

        table1.setRowCount(nr)
        table1.setColumnCount(2)
        table1.setHorizontalHeaderLabels(['Original', 'Selected Features'])

        for r in range(nr):
            item = QTableWidgetItem(self.params.X.columns[r])
            item.setFlags(Qt.ItemIsEnabled)
            table1.setItem(r, 0, item)

            if self.params.X.columns[r] in self.params.X_fs.columns:
                item = QTableWidgetItem(self.params.X.columns[r])
            else:
                item = QTableWidgetItem('')
            item.setFlags(Qt.ItemIsEnabled)
            table1.setItem(r, 1, item)

        table1.setNonScroll()

        self.vbox.addWidget(table1)

        if self.params.lang == 'en':
            text = ('MALSS selected appropriate algorithms '
                    'according to your task and data.\n'
                    'Selected algorithms are shown below.\n'
                    '(Selected algorithms may be changed due to '
                    'the result of feature selection.)')
            self.set_paragraph('Algorithms', text=text)
        else:
            text = ('MALSSは分析タスクとデータに応じて適切なアルゴリズムを選択します．\n'
                    'アルゴリズム選択結果を以下の示します．\n'
                    '（特徴量が減ったことで選ばれるアルゴリズムが変わることがあります．）')
            self.set_paragraph('アルゴリズム', text=text)

        nr = len(self.params.algorithms_fs)

        table2 = NonScrollTable(self.inner)

        table2.setRowCount(nr)
        table2.setColumnCount(1)
        table2.setHorizontalHeaderLabels(['Algorithms'])

        for r in range(nr):
            item = QTableWidgetItem(self.params.algorithms_fs[r][0])
            item.setFlags(Qt.ItemIsEnabled)
            table2.setItem(r, 0, item)

        table2.setNonScroll()

        self.vbox.addWidget(table2)

        if self.params.lang == 'en':
            text = ('Clik "Analyze" to re-analyze with selected features.\n'
                    '(It will take tens of minutes.)')
            self.set_paragraph('', text=text)
        else:
            text = ('それでは特徴量選択した説明変数を用いて分析をしてみましょう，\n'
                    'Anayzeボタンを押してください．\n'
                    '（分析には数分～数十分かかります）')
            self.set_paragraph('', text=text)

        self.vbox.addStretch(1)

        btn = QPushButton('Analyze', self.inner)
        btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            next_page = 'Results 2'
        else:
            next_page = '結果の確認２'
        btn.clicked.connect(lambda: self.button_clicked(
            self.params.mdl_fs, self.params.X_fs, self.params.y, next_page))

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
        if self.params.results_fs is not None:
            self.add_algorithm(self.params.mdl_fs, self.params.algorithms_fs,
                               self.params.results_fs)

    def analyzed(self, signalData):
        self.wait_ani.hide()
        if 'error' in signalData:
            self.params.error = signalData['error']
            self.button_func('Error')
        else:
            if self.params.results_fs is None:
                self.params.results_fs = signalData
            else:
                for name, results in signalData['algorithms'].items():
                    self.params.results_fs['algorithms'][name]['grid_scores'] =\
                        results['grid_scores']
                    self.params.results_fs['algorithms'][name]['learning_curve'] =\
                        results['learning_curve']

            if self.params.lang == 'en':
                self.button_func('Results 2')
            else:
                self.button_func('結果の確認２')
