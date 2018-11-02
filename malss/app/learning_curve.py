# coding: utf-8

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QWidget,
                             QTableWidgetItem, QRadioButton, QButtonGroup,
                             QLabel, QLineEdit, QDoubleSpinBox,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from .content import Content


class LearningCurve(Content):
    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'LearningCurve', params)

        self.button_func = button_func

        ylim = self.__get_ylim(self.params.results['algorithms'])
        for name, val in self.params.results['algorithms'].items():
            self.set_paragraph(h2=name)

            x = val['learning_curve']['x']
            y_train = val['learning_curve']['y_train']
            y_cv = val['learning_curve']['y_cv']
            fig = PlotLearningCurve(x, y_train, y_cv, ylim, name, self.inner)
            self.vbox.addWidget(fig)

        self.vbox.addStretch(1)

    def __get_ylim(self, algorithms):
        ymin = float('Inf')
        ymax = -float('Inf')

        for name, val in algorithms.items():
            ymin = min([ymin, min(val['learning_curve']['y_cv'])])
            ymax = max([ymax, max(val['learning_curve']['y_train'])])
        margin = 0.05 * (ymax - ymin)

        return (ymin - margin, ymax + margin)


class PlotLearningCurve(FigureCanvas):
    def __init__(self, x, y_train, y_cv, ylim, title, parent=None,
                 width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed, QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.plot(x, y_train, y_cv, ylim, title)

    def plot(self, x, y_train, y_cv, ylim, title):
        ax = self.figure.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.grid(True)

        ax.plot(x, y_train, 'o-', color='dodgerblue', label='Training score')
        ax.plot(x, y_cv, 'o-', color='darkorange',
                label='Cross-validation score')
        ax.set_ylim(ylim)
        ax.legend(loc="lower right")

        self.draw()
