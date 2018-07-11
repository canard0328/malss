# coding: utf-8

from PyQt5.QtWidgets import (QScrollArea, QWidget, QVBoxLayout,
        QLabel, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette


class Content(QScrollArea):

    def __init__(self, parent=None, title=''):
        super().__init__(parent)

        self.H1_HEIGHT = 50
        self.H2_HEIGHT = 50 
        self.SIDE_MARGIN = 5
        self.H1_FONT_SIZE = 18
        self.H2_FONT_SIZE = 18
        self.H3_FONT_SIZE = 16
        self.TEXT_FONT_SIZE = 14

        self.inner = QWidget(self)
        
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setSpacing(0)
        self.vbox.setContentsMargins(0, 0, 0, 0)

        topframe = QFrame()
        topframe.setStyleSheet('background-color: white')
        topframe.setFixedHeight(self.H1_HEIGHT)

        lbl_h1 = QLabel(title, topframe)
        fnt = lbl_h1.font()
        fnt.setPointSize(self.H1_FONT_SIZE)
        lbl_h1.setFont(fnt)
        lbl_h1.setFixedHeight(self.H1_HEIGHT)
        lbl_h1.setMargin(self.SIDE_MARGIN)

        self.vbox.addWidget(topframe)

        self.inner.setLayout(self.vbox)

        self.setWidget(self.inner)

    def set_paragraph(self, h2='', h3='', text='', img=None):
        if h2 != '':
            lbl_h2 = QLabel(h2, self.inner)
            fnt = lbl_h2.font()
            fnt.setPointSize(self.H2_FONT_SIZE)
            lbl_h2.setFont(fnt)
            lbl_h2.setFixedHeight(self.H2_HEIGHT)
            lbl_h2.setAlignment(Qt.AlignBottom)
            lbl_h2.setMargin(self.SIDE_MARGIN)
            self.vbox.addWidget(lbl_h2)

            frm = QFrame(self.inner)
            frm.setFrameShape(QFrame.HLine)
            frm.setContentsMargins(self.SIDE_MARGIN, 0, self.SIDE_MARGIN, 0)
            plt = frm.palette()
            plt.setColor(QPalette.WindowText, Qt.darkGray)
            frm.setPalette(plt)
            self.vbox.addWidget(frm)

        if text != '':
            lbl_txt = QLabel(text, self.inner)
            fnt = lbl_txt.font()
            fnt.setPointSize(self.TEXT_FONT_SIZE)
            lbl_txt.setFont(fnt)
            lbl_txt.setMargin(self.SIDE_MARGIN)
            self.vbox.addWidget(lbl_txt)

        self.inner.setLayout(self.vbox)
