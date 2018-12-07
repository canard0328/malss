# coding: utf-8

from PyQt5.QtWidgets import (QScrollArea, QWidget, QVBoxLayout,
                             QLabel, QFrame, QStyle)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPixmap


class Content(QScrollArea):

    def __init__(self, parent=None, title='', params=None):
        super().__init__(parent)

        self.params = params

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.lbl_img_list = []
        self.pixmap_list = []

        self.H1_HEIGHT = 50
        self.H2_HEIGHT = 50
        self.SIDE_MARGIN = 5
        self.H1_FONT_SIZE = 18
        self.H2_FONT_SIZE = 18
        self.H3_FONT_SIZE = 16
        self.TEXT_FONT_SIZE = 14

        self.inner = QWidget(self)

        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setSpacing(10)
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
            lbl_txt.setWordWrap(True)
            fnt = lbl_txt.font()
            fnt.setPointSize(self.TEXT_FONT_SIZE)
            lbl_txt.setFont(fnt)
            lbl_txt.setMargin(self.SIDE_MARGIN)
            self.vbox.addWidget(lbl_txt)

        if img is not None:
            if self.params.lang == 'en':
                img += '_en.png'
            else:
                img += '_jp.png'
            pixmap = QPixmap(img)
            if not pixmap.isNull():
                lbl_img = QLabel(self.inner)
                lbl_img.setPixmap(pixmap)
                self.lbl_img_list.append(lbl_img)
                self.pixmap_list.append(pixmap.scaledToWidth(pixmap.width()))
                self.vbox.addWidget(lbl_img)

        self.inner.setLayout(self.vbox)

    def get_text(self, path):
        if self.params.lang == 'en':
            path += '_en.txt'
        else:
            path += '_jp.txt'

        try:
            text = open(path, encoding='utf8').read()
        except FileNotFoundError:
            text = 'No text available'
        return text

    def make_dtype(self, columns, dtypes):
        if columns is None or dtypes is None:
            return None

        dic = {}
        for c, d in zip(columns, dtypes):
            dic[c] = d
        return dic

    def resizeEvent(self, event):
        # Resize images only if the width of the scroll area
        # is shorter than that of images
        for i, lbl in enumerate(self.lbl_img_list):
            w = self.width() - QStyle.PM_ScrollBarExtent
            if w < self.pixmap_list[i].width():
                lbl.setPixmap(
                        self.pixmap_list[i].scaledToWidth(
                            w, Qt.SmoothTransformation))
            else:
                lbl.setPixmap(
                        self.pixmap_list[i].scaledToWidth(
                            self.pixmap_list[i].width()))
        return super().resizeEvent(event)
