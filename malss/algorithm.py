# -*- coding: utf-8 -*-


class Algorithm(object):
    def __init__(self, estimator, parameters, name):
        self.estimator = estimator
        self.parameters = parameters
        self.best_score = None
        self.name = name
        self.description = '<h2 id="%s">%s ' % (name, name) + \
            '<font size="-1">[<a href="#top">Back To Top</a>]</font></h2>\n'
