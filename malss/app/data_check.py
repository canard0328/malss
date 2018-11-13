# coding: utf-8

import os
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton,
                             QTableWidgetItem, QRadioButton, QButtonGroup,
                             QWidget)
from PyQt5.QtCore import Qt
import pandas as pd
from .content import Content
from .nonscroll_table import NonScrollTable


class DataCheck(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Data check', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        if self.params.data5 is None:
            data = pd.read_csv(
                self.params.fpath, header=0,
                dtype=self.make_dtype(self.params.columns,
                                      self.params.col_types))
            self.params.data5 = data.head(min(5, data.shape[0]))

            self.params.columns = data.columns
            self.params.col_types_def =\
                list(map(str, data.dtypes.get_values()))
            self.params.col_types =\
                list(map(str, data.dtypes.get_values()))

        # nr = min(5, data.shape[0])
        nr = len(self.params.data5)

        if params.lang == 'jp':
            text = ('データの読み込み結果（先頭{n}行）を以下に示します．\n'
                    'データを正しく読み込めていることを確認してください．'.format(n=nr))
            self.set_paragraph(
                'データ読み込み結果の確認', text=text)
        else:
            text = ('First {n} rows of your data are shown below.\n'
                    'Confirm that the data was read correctly.'.format(n=nr))
            self.set_paragraph(
                'Check data', text=text)

        table = NonScrollTable(self.inner)

        table.setRowCount(nr)
        table.setColumnCount(len(self.params.columns))
        table.setHorizontalHeaderLabels(self.params.columns)

        for r in range(nr):
            for c in range(len(self.params.columns)):
                item = QTableWidgetItem(str(self.params.data5.iat[r, c]))
                item.setFlags(Qt.ItemIsEnabled)
                table.setItem(r, c, item)

        table.setNonScroll()

        self.vbox.addWidget(table)

        # Text for type checking and objective variable setting
        path1 = path + 'categorical'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph(
                'Check type of variables and set objective variable',
                text=text)
        else:
            self.set_paragraph('変数タイプの確認と目的変数の設定', text=text)

        # htable = QTableWidget(self.inner)
        htable = NonScrollTable(self.inner)

        htable.setRowCount(len(self.params.columns))
        htable.setColumnCount(4)
        htable.setHorizontalHeaderLabels(
            ['columns', 'categorical', 'quantitative', 'objective variable'])

        self.lst_cat = []
        self.lst_num = []
        self.lst_obj = []
        self.obj_group = QButtonGroup(self.inner)
        for c in range(len(self.params.columns)):
            # col 1
            item = QTableWidgetItem(self.params.columns[c])
            item.setFlags(Qt.ItemIsEnabled)
            htable.setItem(c, 0, item)

            group = QButtonGroup(self.inner)

            # col 2
            htable.setCellWidget(
                c, 1,
                self.__make_cell(c, 'cat', self.params.col_types[c],
                                 self.params.col_types_def[c]))
            group.addButton(self.lst_cat[-1])

            # col 3
            htable.setCellWidget(
                c, 2,
                self.__make_cell(c, 'num', self.params.col_types[c],
                                 self.params.col_types_def[c]))
            group.addButton(self.lst_num[-1])

            # col 4
            htable.setCellWidget(
                c, 3,
                self.__make_cell(c, 'obj', self.params.col_types[c],
                                 self.params.col_types_def[c]))
            self.obj_group.addButton(self.lst_obj[-1])
            self.obj_group.setId(self.lst_obj[-1], c)

        htable.setNonScroll()

        self.vbox.addWidget(htable)

        hbox2 = QHBoxLayout()
        hbox2.setContentsMargins(10, 10, 10, 10)

        self.btn = QPushButton('Next', self.inner)
        if self.params.lang == 'en':
            self.btn.clicked.connect(lambda: self.button_func('Overfitting'))
        else:
            self.btn.clicked.connect(lambda: self.button_func('過学習'))
        if self.obj_group.checkedButton() is None:
            self.btn.setEnabled(False)
        else:
            self.btn.setEnabled(True)

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn)

        self.vbox.addLayout(hbox2)

        self.vbox.addStretch(1)

    def __make_cell(self, c, name, col_type, col_type_def):
        cell = QWidget(self.inner)
        rbtn = QRadioButton('', cell)
        rbtn.toggled.connect(lambda: self.rbtn_clicked(name + '_' + str(c)))
        hbl = QHBoxLayout(cell)
        hbl.addWidget(rbtn)
        hbl.setContentsMargins(0, 0, 0, 0)
        hbl.setAlignment(Qt.AlignCenter)
        cell.setLayout(hbl)
        if name == 'cat':
            if col_type == 'object':
                rbtn.setChecked(True)
            self.lst_cat.append(rbtn)
        elif name == 'num':
            if col_type != 'object':
                rbtn.setChecked(True)
            if col_type_def == 'object':
                rbtn.setEnabled(False)
            self.lst_num.append(rbtn)
        elif name == 'obj':
            if col_type == 'object' and self.params.task == 'Regression':
                rbtn.setEnabled(False)
            elif col_type != 'object' and self.params.task == 'Classification':
                rbtn.setEnabled(False)
            if self.params.columns[c] == self.params.objective:
                rbtn.setChecked(True)
            self.lst_obj.append(rbtn)

        return cell

    def rbtn_clicked(self, text):
        name, idx = text.split('_')
        idx = int(idx)
        if len(self.lst_obj) <= idx:
            return

        if self.lst_num[idx].isChecked():
            if self.params.task == 'Classification':
                self.obj_group.setExclusive(False)
                self.lst_obj[idx].setChecked(False)
                self.obj_group.setExclusive(True)
                self.lst_obj[idx].setEnabled(False)
            elif self.params.task == 'Regression':
                self.lst_obj[idx].setEnabled(True)

            self.params.col_types[idx] = self.params.col_types_def[idx]
        elif self.lst_cat[idx].isChecked():
            if self.params.task == 'Classification':
                self.lst_obj[idx].setEnabled(True)
            elif self.params.task == 'Regression':
                self.obj_group.setExclusive(False)
                self.lst_obj[idx].setChecked(False)
                self.obj_group.setExclusive(True)
                self.lst_obj[idx].setEnabled(False)

            self.params.col_types[idx] = 'object'

        if self.obj_group.checkedButton() is None:
            self.params.objective = None
            self.btn.setEnabled(False)
        else:
            self.params.objective =\
                self.params.columns[self.obj_group.checkedId()]
            self.btn.setEnabled(True)

        self.params.col_types_changed = True
