# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QLineEdit, QTableWidgetItem)
from PyQt5.QtCore import Qt
from .content import Content
from .nonscroll_table import NonScrollTable


class Prediction(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Prediction', params)

        self.button_func = button_func

        if params.lang == 'jp':
            text=('最も性能のよいモデルは以下のアルゴリズム，'
                  'ハイパーパラメータでした．')
            if self.params.not_deleted:
                text += '\n（特徴量選択の結果，説明変数は削減されませんでした．）'
            self.set_paragraph('最良のモデル', text=text)
        else:
            text=('Algorithm and hyper-parameters of the best model '
                  'are shown below.')
            if self.params.not_deleted:
                text += '\n(No features were deleted by feature selection.)'
            self.set_paragraph('Best model', text=text)

        self.show_best_algorithm()

        if params.lang == 'jp':
            self.set_paragraph(
                '学習結果の出力',
                text=('学習結果をresultsというフォルダ内に出力します．\n'
                      '予測モデルを出力するディレクトリを選択してください。\n'
                      '同名のファイルがある場合上書きされますので気をつけてください．'))
        else:
            self.set_paragraph(
                'Output your results',
                text=('You can output your results in the folder named'
                      '"results".\n'
                      'Choose your output directory.\n'
                      'Note that tha file will be overwritten.'))

        hbox1 = QHBoxLayout()
        hbox1.setContentsMargins(10, 10, 10, 10)

        lbl_dir = QLabel('Location:', self.inner)
        self.le_dir = QLineEdit(self.inner)
        do_btn = QPushButton('Browse...', self.inner)
        do_btn.clicked.connect(self.dir_open_dialog)

        hbox1.addWidget(lbl_dir)
        hbox1.addWidget(self.le_dir)
        hbox1.addWidget(do_btn)

        self.vbox.addLayout(hbox1)

        if self.params.out_dir is not None:
            self.le_dir.setText(self.params.out_dir)

        if params.lang == 'jp':
            self.set_paragraph(
                '未知データの予測',
                text=('学習したモデルで予測を行いたいデータ（目的変数の無いデータ）'
                      'があればファイルを選択してください．\n'
                      '入力ファイルは学習用データと説明変数の並び順が同じで，'
                      '目的変数の無いCSVファイルである必要があります\n'
                      '予測結果は入力ファイルと同じディレクトリにpredict.csv'
                      'という名前で出力されます．同名のファイルは上書きされます'
                      'ので気をつけてください．'))
        else:
            self.set_paragraph(
                'Prediction',
                text=('If you have a data to predict target variable, '
                      'choose your input file.\n'
                      'Input file must have same features as training data '
                      '(same order) and must not have target variable.\n'
                      'The results save as predict.csv in the same directory '
                      'as the input file. Note that the file will be '
                      'overwritten.'))

        hbox2 = QHBoxLayout()
        hbox2.setContentsMargins(10, 10, 10, 10)

        lbl_file = QLabel('Location:', self.inner)
        self.le_file = QLineEdit(self.inner)
        fo_btn = QPushButton('Browse...', self.inner)
        fo_btn.clicked.connect(self.file_open_dialog)

        hbox2.addWidget(lbl_file)
        hbox2.addWidget(self.le_file)
        hbox2.addWidget(fo_btn)

        self.vbox.addLayout(hbox2)

        if self.params.path_pred is not None:
            self.le_file.setText(self.params.path_pred)

        hbox3 = QHBoxLayout()
        hbox3.setContentsMargins(10, 10, 10, 10)

        self.btn = QPushButton('Output', self.inner)
        self.btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        self.btn.clicked.connect(self.__button_clicked)
        self.btn.setEnabled(False)

        hbox3.addStretch(1)
        hbox3.addWidget(self.btn)

        self.vbox.addLayout(hbox3)

        self.vbox.addStretch(1)

    def file_open_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'File open', '')
        self.le_file.setText(fname[0])
        self.params.path_pred = fname[0]

    def dir_open_dialog(self):
        dname = QFileDialog.getExistingDirectory(self, 'Directory open', '')
        dname += '/results'
        self.le_dir.setText(dname)
        self.btn.setEnabled(True)
        self.params.out_dir = dname

    def show_best_algorithm(self):
        self.best_algo_name = None
        self.best_param_names = None
        self.best_params = None
        self.best_score = -float('Inf')
        self.best_is_fs = False
        for name, results in self.params.results['algorithms'].items():
            grid_scores = results['grid_scores']
            for params, score, std in grid_scores:
                if score > self.best_score:
                    self.best_algo_name = name
                    self.best_param_names = list(params.keys())
                    self.best_params = list(params.values())
                    self.best_score = score
        if self.params.results_fs is not None:
            for name, results in self.params.results_fs['algorithms'].items():
                grid_scores = results['grid_scores']
                for params, score, std in grid_scores:
                    if score > self.best_score:
                        self.best_algo_name = name
                        self.best_param_names = list(params.keys())
                        self.best_params = list(params.values())
                        self.best_score = score
                        self.best_is_fs = True

        table1 = NonScrollTable(self.inner)

        table1.setRowCount(1)
        table1.setColumnCount(len(self.best_params) + 3)
        table1.setHorizontalHeaderLabels(
            ['Algorithm', 'Feature selection'] + self.best_param_names +
            ['Score'])

        item = QTableWidgetItem(self.best_algo_name)
        item.setFlags(Qt.ItemIsEnabled)
        table1.setItem(0, 0, item)

        if self.best_is_fs:
            item = QTableWidgetItem('Yes')
        else:
            item = QTableWidgetItem('No')
        item.setFlags(Qt.ItemIsEnabled)
        table1.setItem(0, 1, item)

        for c, v in enumerate(self.best_params):
            item = QTableWidgetItem(str(round(v, 4)))
            item.setFlags(Qt.ItemIsEnabled)
            table1.setItem(0, c + 2, item)

        item = QTableWidgetItem(str(round(self.best_score, 4)))
        item.setFlags(Qt.ItemIsEnabled)
        table1.setItem(0, len(self.best_params) + 2, item)

        table1.setNonScroll()

        self.vbox.addWidget(table1)

    def __button_clicked(self):
        self.__output()

        if self.params.path_pred is not None:
            self.__predict()

    def __output(self):
        if not os.path.exists(self.params.out_dir):
            os.mkdir(self.params.out_dir)

        with open(self.params.out_dir + '/result.txt', 'w') as fo:
            fo.write(
                'Training file: ' + os.path.basename(self.params.fpath) + '\n')
            fo.write(
                'Sample size: ' + str(len(self.params.X)) + '\n')
            fo.write('Columns (types):\n')
            for i, col in enumerate(self.params.columns):
                fo.write('  ' + col + ' (' + self.params.col_types[i] + ')\n')

            fo.write('\nBest result:\n')
            fo.write('  Feature selection: ')
            fo.write('Yes\n') if self.best_is_fs else fo.write('No\n')
            fo.write('  Features:\n')
            X = self.params.X_fs if self.best_is_fs else self.params.X
            for col in X.columns:
                fo.write('    ' + col + '\n')
            fo.write('  Algorithm: ' + self.best_algo_name + '\n')
            fo.write('  Parameters:\n')
            for i, name in enumerate(self.best_param_names):
                fo.write(
                    '    ' + name + ': ' + str(self.best_params[i]) + '\n')
            fo.write('  Score: ' + str(self.best_score) + '\n')

            fo.write('\nAll results\n')
            fo.write('  Feature selection: No\n')
            fo.write('    Features:\n')
            for col in self.params.X.columns:
                fo.write('      ' + col + '\n')
            fo.write('    Algorithms:\n')
            for name, rslt in self.params.results['algorithms'].items():
                self.__write_result(fo, name, rslt)

            if self.params.results_fs is not None:
                fo.write('\n  Feature selection: Yes\n')
                fo.write('    Features:\n')
                for col in self.params.X_fs.columns:
                    fo.write('      ' + col + '\n')
                fo.write('    Algorithms:\n')
                for name, rslt in self.params.results_fs['algorithms'].items():
                    self.__write_result(fo, name, rslt, 'Feature selected')

        self.__plot_curves()

    def __write_result(self, fo, name, result, suffix_lc=''):
        fo.write('      ' + name + '\n')
        fo.write('        Grid scores:\n')
        for pname in result['grid_scores'][0][0].keys():
            fo.write('          ' + pname + '\t')
        fo.write('Score\n')
        for params, score, std in result['grid_scores']:
            for val in params.values():
                fo.write('          ' + str(round(val, 6)) + '\t')
            fo.write(str(round(score, 4)) + '\n')
        fname = 'Learning curve_' + name
        if suffix_lc != '':
            fname += '_' + suffix_lc
        fname += '.png'
        fo.write('        Learning curve: ' + fname + '\n')

    def __plot_curves(self):
        ylim = self.__get_ylim(self.params.results['algorithms'])
        if self.params.results_fs is not None:
            ylim2 = self.__get_ylim(self.params.results_fs['algorithms'])
            if ylim2[0] < ylim[0]:
                ylim[0] = ylim2[0]
            if ylim2[1] > ylim[1]:
                ylim[1] = ylim2[1]

        for name, val in self.params.results['algorithms'].items():
            self.__plot_curve(name, val, ylim)

        if self.params.results_fs is not None:
            for name, val in self.params.results_fs['algorithms'].items():
                self.__plot_curve(name, val, ylim, 'Feature selected')

    def __plot_curve(self, name, val, ylim, suffix=''):
        x = val['learning_curve']['x']
        y_train = val['learning_curve']['y_train']
        y_cv = val['learning_curve']['y_cv']

        plt.plot(x, y_train, 'o-', color='dodgerblue',
                 label='Training score')
        plt.plot(x, y_cv, 'o-', color='darkorange',
                 label='Cross-validation score')
        plt.title(name)
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.grid(True)
        plt.ylim(ylim)
        plt.legend(loc="lower right")
        fname = self.params.out_dir + '/Learning curve_' + name
        if suffix != '':
            fname += '_' + suffix
        fname += '.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    def __get_ylim(self, algorithms):
        ymin = float('Inf')
        ymax = -float('Inf')

        for name, val in algorithms.items():
            ymin = min([ymin, min(val['learning_curve']['y_cv'])])
            ymax = max([ymax, max(val['learning_curve']['y_train'])])
        margin = 0.05 * (ymax - ymin)

        return [ymin - margin, ymax + margin]

    def __predict(self):
        columns = []
        col_types = []
        for i in range(len(self.params.columns)):
            if self.params.columns[i] != self.params.objective:
                columns.append(self.params.columns[i])
                col_types.append(self.params.col_types[i])
        try:
            # engine='python' is to avoid pandas's bug.
            X = pd.read_csv(self.params.path_pred, header=0, engine='python',
                            dtype=self.make_dtype(columns, col_types))
        except Exception:
            import traceback
            self.params.error = traceback.format_exc()
            self.button_func('Error')
            return

        if self.best_is_fs:
            estimator = self.params.results_fs['best_algorithm']['estimator']
            mdl = self.params.mdl_fs
        else:
            estimator = self.params.results['best_algorithm']['estimator']
            mdl = self.params.mdl
        pred = mdl.predict(X, estimator)
        pd.Series(pred).to_csv(
            os.path.dirname(self.params.path_pred) + '/predict.csv',
            index=False)
