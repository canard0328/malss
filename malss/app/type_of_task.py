# coding: utf-8

import os
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton,
                             QRadioButton, QButtonGroup)
from .content import Content


class TypeOfTask(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Task', params)

        self.button_func = button_func

        path = os.path.abspath(os.path.dirname(__file__)) + '/static/'

        # Text for machine learning tasks
        path1 = path + 'task'
        text = self.get_text(path1)
        if self.params.lang == 'en':
            self.set_paragraph('Type of task', text=text, img=path1)
        else:
            self.set_paragraph('分析タスク', text=text, img=path1)

        # Text for supervised learning
        path2 = path + 'supervised_learning'
        text = self.get_text(path2)
        if self.params.lang == 'en':
            self.set_paragraph('Supervised learning', text=text, img=path2)
        else:
            self.set_paragraph('教師あり学習', text=text, img=path2)

        if params.lang == 'jp':
            self.set_paragraph(
                'タスクの選択',
                text='あなたの機械学習のタスクを選択してください。')
        else:
            self.set_paragraph(
                'Task selection',
                text='Choose your machine learning taks.')

        vbox1 = QVBoxLayout()
        vbox1.setContentsMargins(10, 10, 10, 10)
        vbox1.setSpacing(10)

        rbtn_cls = QRadioButton('Classification', self.inner)
        rbtn_cls.setStyleSheet('QRadioButton{font: 15pt;};')
        rbtn_cls.clicked.connect(self.rbtn_clicked)
        rbtn_reg = QRadioButton('Regression', self.inner)
        rbtn_reg.setStyleSheet('QRadioButton{font: 15pt;};')
        rbtn_reg.clicked.connect(self.rbtn_clicked)
        if params.task == 'Classification':
            rbtn_cls.setChecked(True)
        elif params.task == 'Regression':
            rbtn_reg.setChecked(True)

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(rbtn_cls, 1)
        self.btn_group.addButton(rbtn_reg, 1)

        vbox1.addWidget(rbtn_cls)
        vbox1.addWidget(rbtn_reg)
        vbox1.addStretch(1)

        self.vbox.addLayout(vbox1)

        self.vbox.addStretch(1)

        self.btn = QPushButton('Next', self.inner)
        self.btn.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
        if self.params.lang == 'en':
            self.btn.clicked.connect(lambda: self.button_func('Input data'))
        else:
            self.btn.clicked.connect(lambda: self.button_func('入力データ'))
        if params.task is None:
            self.btn.setEnabled(False)

        self.vbox.addWidget(self.btn)

    def rbtn_clicked(self):
        self.params.task = self.btn_group.checkedButton().text()
        self.btn.setEnabled(True)
