# coding: utf-8

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton,
                             QRadioButton, QButtonGroup)
from .content import Content


class TypeOfTask(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Task', params)

        self.button_func = button_func

        if params.lang == 'jp':
            self.set_paragraph(
                'Type of task',
                text='あなたの機械学習のタスクを選択してください。')
        else:
            self.set_paragraph(
                'Type of task',
                text='Choose your machine learning taks.')

        hbox1 = QHBoxLayout()
        hbox1.setContentsMargins(10, 10, 10, 10)

        rbtn_cls = QRadioButton('Classification', self.inner)
        rbtn_cls.clicked.connect(self.rbtn_clicked)
        rbtn_reg = QRadioButton('Regression', self.inner)
        rbtn_reg.clicked.connect(self.rbtn_clicked)
        if params.task == 'Classification':
            rbtn_cls.setChecked(True)
        elif params.task == 'Regression':
            rbtn_reg.setChecked(True)

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(rbtn_cls, 1)
        self.btn_group.addButton(rbtn_reg, 1)

        hbox1.addWidget(rbtn_cls)
        hbox1.addWidget(rbtn_reg)
        hbox1.addStretch(1)

        self.vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        hbox2.setContentsMargins(10, 10, 10, 10)

        self.btn = QPushButton('Next', self.inner)
        self.btn.clicked.connect(lambda: self.button_func('File selection'))
        if params.task is None:
            self.btn.setEnabled(False)

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn)

        self.vbox.addLayout(hbox2)

        self.vbox.addStretch(1)

    def rbtn_clicked(self):
        self.params.task = self.btn_group.checkedButton().text()
        self.btn.setEnabled(True)
