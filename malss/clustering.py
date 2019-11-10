from sklearn.cluster import KMeans

from .algorithm import Algorithm

class Clustering(object):
    @staticmethod
    def choose_algorithm(min_clusters, max_clusters):
        algorithms = []
        algorithms.append(
            Algorithm(
                KMeans(),
                [{'n_clusters': list(range(min_clusters, max_clusters+1))}],
                'K-Means',
                ('https://scikit-learn.org/stable/modules/generated/'
                    'sklearn.cluster.KMeans.html')))
        
        return algorithms