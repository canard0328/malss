# -*- coding: utf-8 -*-


class Algorithm(object):
    def __init__(self, estimator, parameters, name):
        self.estimator = estimator
        self.parameters = parameters
        self.best_score = None
        self.name = name
        self.description = '<h2>%s</h2>\n' % name
