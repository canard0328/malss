import numpy as np
import pandas
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

from .algorithm import Algorithm

class Clustering(object):
    @staticmethod
    def choose_algorithm(min_clusters, max_clusters, random_state):
        algorithms = []
        algorithms.append(
            Algorithm(
                KMeans(random_state=random_state),
                [{'n_clusters': list(range(min_clusters, max_clusters+1))}],
                'K-Means',
                ('https://scikit-learn.org/stable/modules/generated/'
                    'sklearn.cluster.KMeans.html')))
        
        return algorithms
    
    @classmethod
    def analyze(cls, algorithms, data, min_clusters, max_clusters, random_state, verbose):
        for i in range(len(algorithms)):
            if verbose:
                print('    %s' % algorithms[i].name)
            if isinstance(data.X, pandas.DataFrame):
                X = data.X.to_numpy()
            else:
                X = data.X
            gap = Clustering.calc_gap(algorithms[i].estimator, X, min_clusters, max_clusters, random_state)
    
    @staticmethod
    def calc_inertia(a, X):
        W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
        return np.mean(W)
    
    @classmethod
    def calc_gap(cls, algorithm, X, min_clusters, max_clusters, random_state):
        np.random.seed(random_state)
        inertia_data = []
        inertia_ref = []
        for nc in range(min_clusters, max_clusters + 1):
            algorithm.n_clusters = nc

            pred_labels = algorithm.fit_predict(X)
            inertia_data.append(Clustering.calc_inertia(pred_labels, X))

            inertia_ref_sub = []
            for _ in range(5):
                ref = np.random.rand(*X.shape)
                ref = (ref * (X.max(axis=0) - X.min(axis=0))) + X.min(axis=0)
                pred_labels = algorithm.fit_predict(ref)
                inertia_ref_sub.append(Clustering.calc_inertia(pred_labels, ref))
            inertia_ref.append(np.mean(inertia_ref_sub))
        
        return np.log(inertia_ref) - np.log(inertia_data)