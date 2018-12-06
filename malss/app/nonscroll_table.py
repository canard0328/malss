# coding: utf-8

from PyQt5.QtWidgets import QTableWidget, QHeaderView
from PyQt5.QtCore import Qt, QSize


class NonScrollTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.header = self.horizontalHeader()
        self.header.setSectionResizeMode(QHeaderView.ResizeToContents)

    def setNonScroll(self):
        # Only scrollable for horizontal axis.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMaximumSize(self._getQTableWidgetSize())
        self.setMinimumHeight(self._getQTableWidgetHeight())

    def _getQTableWidgetSize(self):
        w = self._getQTableWidgetWidth()
        h = self._getQTableWidgetHeight()
        return QSize(w, h)

    def _getQTableWidgetWidth(self):
        w = self.verticalHeader().width() + 4  # +4 seems to be needed
        for i in range(self.columnCount()):
            w += self.columnWidth(i)  # seems to include gridline
        return w

    def _getQTableWidgetHeight(self):
        h = self.horizontalHeader().height() + 4
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        return h