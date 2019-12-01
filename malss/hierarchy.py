# -*- coding: utf-8 -*-
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


class HierarchicalClustering(object):
    def __init__(self, n_clusters=3, random_state=None, method='complete', metric='euclidean'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.method = method
        self.metric = metric

    def fit_predict(self, X, y=None):
        self.model = linkage(X, method=self.method, metric=self.metric)
        return fcluster(self.model, t=self.n_clusters, criterion='maxclust') - 1