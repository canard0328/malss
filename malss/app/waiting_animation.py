# coding: utf-8

import os
import math
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPainter, QBrush, QColor, QPen
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPalette, QPixmap, QImage


class WaitingAnimation(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.lists = None
        self.list_index = 0

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 200)))
        painter.setPen(QPen(Qt.NoPen))

        if self.lists is not None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/static/'
            path += self.lists[self.list_index] + '.png'
            self.list_index += 1
            if self.list_index >= len(self.lists):
                self.list_index = 0
            image = QImage(path)
            rect_image = image.rect()
            rect_painter = event.rect()
            dx = (rect_painter.width() - rect_image.width()) / 2.0
            dy = (rect_painter.height() - rect_image.height()) / 2.0
            painter.drawImage(dx, dy, image)

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(5000)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()
    
    def set_lists(self, lists):
        self.lists = lists
