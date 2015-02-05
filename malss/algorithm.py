# -*- coding: utf-8 -*-


class Algorithm(object):
    def __init__(self, estimator, parameters, name, link=None):
        self.estimator = estimator
        self.parameters = parameters
        self.best_score = None
        self.best_params = None
        self.is_best_algorithm = False
        self.grid_scores = None
        self.classification_report = None
        self.name = name
        self.link = link
