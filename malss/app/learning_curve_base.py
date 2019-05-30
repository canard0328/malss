# coding: utf-8

from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .content import Content


class LearningCurveBase(Content):
    def __init__(self, parent=None, title='', params=None):
        super().__init__(parent, title, params)

    def plot_curve(self, algo_results):
        ylim = self.__get_ylim(algo_results)
        for name, val in algo_results.items():
            self.set_paragraph(h2=name)

            x = val['learning_curve']['x']
            y_train = val['learning_curve']['y_train']
            y_cv = val['learning_curve']['y_cv']
            fig = PlotLearningCurve(x, y_train, y_cv, ylim, name, self.inner)
            self.vbox.addWidget(fig)

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
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed, QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.plot(x, y_train, y_cv, ylim, title)

    def plot(self, x, y_train, y_cv, ylim, title):
        self.axes.set_title(title)
        self.axes.set_xlabel('Training examples')
        self.axes.set_ylabel('Score')
        self.axes.grid(True)

        self.axes.plot(x, y_train, 'o-', color='dodgerblue', label='Training score')
        self.axes.plot(x, y_cv, 'o-', color='darkorange',
                       label='Cross-validation score')
        self.axes.set_ylim(ylim)
        self.axes.legend(loc="lower right")

        self.draw()
