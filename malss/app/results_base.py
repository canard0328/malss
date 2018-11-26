# coding: utf-8

import os
from PyQt5.QtWidgets import (QHBoxLayout, QTableWidgetItem,
                             QLabel, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor
import numpy as np
from .content import Content
from .nonscroll_table import NonScrollTable


class ResultsBase(Content):
    def __init__(self, parent=None, title='', params=None):
        super().__init__(parent, title, params)

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        path1 = path + 'results'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('', text=text)
        else:
            self.set_paragraph('', text=text)

        self.sb_list_from = []
        self.sb_list_to = []
        self.sb_list_div = []

    def __make_spinbox(self, algorithms, name, param_name, idx, min_, max_,
                       val, step, prefix=None, is_double=False):
        if is_double:
            sb = QDoubleSpinBox(self.inner)
        else:
            sb = QSpinBox(self.inner)
        sb.setRange(min_, max_)
        sb.setValue(val)
        sb.setSingleStep(step)
        if prefix is not None:
            sb.setPrefix(prefix)
        sb.setFocusPolicy(Qt.StrongFocus)
        sb.installEventFilter(self)
        sb.valueChanged.connect(lambda: self.__sb_changed(
            algorithms, name + '=' + param_name + '=' + str(idx)))

        return sb

    def __sb_changed(self, algorithms, text):
        algo_name, param_name, idx = text.split('=')
        idx = int(idx)
        from_ = self.sb_list_from[idx].value()
        to = self.sb_list_to[idx].value()
        div = self.sb_list_div[idx].value()
        prefix = self.sb_list_from[idx].prefix()

        from_to = np.linspace(from_, to, div)
        if prefix == '1e':
            from_to = list(10 ** from_to)
        elif prefix == '':
            from_to = list(from_to)
        else:
            raise Exception('Wrong prefix.')

        for name, params in algorithms:
            if name == algo_name:
                if param_name in params[0]:
                    params[0][param_name] = from_to

        self.btn_next.setEnabled(False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and 'SpinBox' in str(obj):
            return True

        return False

    def _check_param(self, name, values):
        if name == 'alpha' or name == 'C' or name == 'gamma':
            return ('1e', np.log10(values.min()), np.log10(values.max()),
                    len(set(values)), -6.0, 6.0, 0.25)
        elif name == 'max_depth':
            return ('', values.min(), values.max(), len(set(values)), 1, 15, 1)
        elif name == 'max_features':
            return ('', values.min(), values.max(), len(set(values)),
                    0.1, 1.0, 0.1)
        elif name == 'n_neighbors':
            return ('', values.min(), values.max(), len(set(values)), 2, 20, 1)
        else:
            raise Exception('Unknown hyper parameter {x}'.format(x=name))

    def make_tables(self, algo_result, algo_param):
        idx_algo = 0
        for name, val in algo_result.items():
            self.set_paragraph(h2=name)

            grid_scores = val['grid_scores']
            param_names = []
            scores = []
            best_score = -float('inf')
            for (params, score, std) in grid_scores:
                if len(param_names) == 0:
                    param_names = list(params.keys())
                score = np.round(score, decimals=6)
                if best_score < score:
                    best_score = score
                scores.append([params[p] for p in param_names] + [score])

            table = NonScrollTable(self.inner)

            table.setRowCount(len(scores))
            table.setColumnCount(len(scores[0]))
            table.setHorizontalHeaderLabels(param_names + ['Score'])
            table.setNonScroll()

            for r in range(len(scores)):
                for c in range(len(scores[0])):
                    item = QTableWidgetItem(str(scores[r][c]))
                    item.setFlags(Qt.ItemIsEnabled)
                    table.setItem(r, c, item)
                    if scores[r][-1] == best_score:
                        table.item(r, c).setBackground(QColor('#ffeeee'))

            self.vbox.addWidget(table)

            for idx, param_name in enumerate(param_names):
                hbox = QHBoxLayout()

                lbl = QLabel(param_name + ': ', self.inner)

                prefix, from_, to, div, min_, max_, step = self._check_param(
                        param_name, np.array(scores)[:, idx])

                lbl_from = QLabel('from:')
                self.sb_list_from.append(
                        self.__make_spinbox(algo_param,
                                            name, param_name, idx_algo, min_,
                                            max_, from_, step, prefix, True))

                lbl_to = QLabel(' to:')
                self.sb_list_to.append(
                        self.__make_spinbox(algo_param,
                                            name, param_name, idx_algo, min_,
                                            max_, to, step, prefix, True))

                lbl_div = QLabel(' divides into:')
                self.sb_list_div.append(
                        self.__make_spinbox(algo_param,
                                            name, param_name, idx_algo, 2,
                                            9, div, 1))

                hbox.addWidget(lbl)
                hbox.addWidget(lbl_from)
                hbox.addWidget(self.sb_list_from[-1])
                hbox.addWidget(lbl_to)
                hbox.addWidget(self.sb_list_to[-1])
                hbox.addWidget(lbl_div)
                hbox.addWidget(self.sb_list_div[-1])

                hbox.addStretch()

                self.vbox.addLayout(hbox)

                idx_algo += 1
