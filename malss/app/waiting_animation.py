# coding: utf-8

import math
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPainter, QBrush, QColor, QPen
from PyQt5.QtWidgets import QWidget


class WaitingAnimation(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 200)))
        painter.setPen(QPen(Qt.NoPen))

        for i in range(6):
            if self.counter % 6 == i:
                painter.setBrush(QBrush(QColor(204, 112, 0)))
            else:
                painter.setBrush(QBrush(QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 +
                40 * math.cos((2 * math.pi * i - math.pi) / 6.0) - 10,
                self.height()/2 +
                40 * math.sin((2 * math.pi * i - math.pi) / 6.0) - 10,
                20, 20)

        painter.setPen(QColor(64, 64, 64))
        painter.drawText(self.rect(), Qt.AlignCenter, 'Processing...')

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(1000)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()
