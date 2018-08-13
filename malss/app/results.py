# coding: utf-8

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QTableWidget,
                             QTableWidgetItem, QRadioButton, QButtonGroup,
                             QWidget)
from PyQt5.QtCore import Qt
import pandas as pd
from .content import Content


class Results(Content):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Results', params)

        algorithms = self.params.results['algorithms']
        for name, scores in algorithms.items():
            pass
