# coding: utf-8

import pandas as pd
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from ..malss import MALSS
from .content import Content
from multiprocessing import Process, Queue
from .waiting_animation import WaitingAnimation
from .nonscroll_table import NonScrollTable


class FeatureSelection(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Feature selection', params)

        self.button_func = button_func

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
