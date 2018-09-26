# coding: utf-8

from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtCore import Qt, QSize


class NonScrollTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setNonScroll(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMaximumSize(self._getQTableWidgetSize())
        self.setMinimumSize(self._getQTableWidgetSize())

    def _getQTableWidgetSize(self):
        w = self.verticalHeader().width() + 4  # +4 seems to be needed
        for i in range(self.columnCount()):
            w += self.columnWidth(i)  # seems to include gridline
        h = self.horizontalHeader().height() + 4
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        return QSize(w, h)
