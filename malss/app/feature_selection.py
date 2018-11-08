# coding: utf-8

import pandas as pd
from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal
from ..malss import MALSS
from .content import Content
from multiprocessing import Process, Queue
from .waiting_animation import WaitingAnimation


class FeatureSelection(Content):

    def __init__(self, parent=None, button_func=None, params=None):
        super().__init__(parent, 'Feature selection', params)

        self.button_func = button_func
