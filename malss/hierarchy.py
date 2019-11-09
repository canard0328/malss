# -*- coding: utf-8 -*-
from scipy.cluster.hierarchy import linkage, dendrogram


class HierarchicalClustering(object):
    def __init__(self, method='average', metric='euclidean'):
        self.method = method
        self.metric = metric

    def fit_predict(self, X, y=None):
        self.Z = linkage(X, method=self.method, metric=self.metric)