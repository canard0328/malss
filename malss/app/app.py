# coding: utf-8

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QApplication, QFrame,
        QVBoxLayout, QSplitter, QDesktopWidget)
from .introduction import Introduction
from .analysis import Analysis
from .menuview import MenuView


class App(QWidget):
    def __init__(self, lang='en'):
        super().__init__()
        self.lang = lang
        self.initUI()

    
    def initUI(self):
        self.txt2func = {'Introduction': Introduction, 'Analysis': Analysis}

        self.setMinimumSize(900, 600)
        self.setStyleSheet('background-color: rgb(242, 242, 242)')

        vbox = QVBoxLayout(self)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        top = QFrame(self)
        top.setFrameShape(QFrame.StyledPanel)
        top.setFixedHeight(50)
        top.setStyleSheet('background-color: white')

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setHandleWidth(0)

        self.menuview = MenuView(self.splitter, self.update_content, self.lang)
        self.menuview.setWidgetResizable(True)

        self.contentview = Introduction(self.splitter, self.menuview.add_button, self.lang)
        self.contentview.setWidgetResizable(True)

        self.splitter.addWidget(self.menuview)
        self.splitter.addWidget(self.contentview)

        vbox.addWidget(top)
        vbox.addWidget(self.splitter)

        self.setLayout(vbox)
        
        self.center()
        self.setWindowTitle('MALSS interactive')
        self.show()

    def center(self):
        # Get a rectangle of the main window.
        qr = self.frameGeometry()
        # Figure out the screen resolution; and from this resolution, get the center point (x, y)
        cp = QDesktopWidget().availableGeometry().center()
        # Set the center of the rectangle to the center of the screen.
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_content(self, text):
        content = self.splitter.widget(1)
        if content is not None:
            if text in self.txt2func:
                content.hide()
                content.deleteLater()

                self.contentview = self.txt2func[text](self.splitter,
                                                       self.menuview.add_button)
                self.contentview.setWidgetResizable(True)
                self.splitter.addWidget(self.contentview)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
