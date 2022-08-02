import os
import io
import shutil
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from jinja2 import Environment, FileSystemLoader

from .algorithm import Algorithm
from .hierarchy import HierarchicalClustering


class Clustering(object):
    @staticmethod
    def choose_algorithm(min_clusters, max_clusters, random_state):
        algorithms = []
        algorithms.append(
            Algorithm(
                KMeans(random_state=random_state),
                [],
                'K-Means',
                ('https://scikit-learn.org/stable/modules/generated/'
                 'sklearn.cluster.KMeans.html')
            )
        )
        algorithms.append(
            Algorithm(
                HierarchicalClustering(random_state=random_state),
                [],
                'Hierarchical Clustering',
                ('https://docs.scipy.org/doc/scipy/reference/generated/'
                 'scipy.cluster.hierarchy.linkage.html')
            )
        )
        
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
            
            gap, sk, nc = Clustering.calc_gap(algorithms[i].estimator, X, min_clusters,
                                              max_clusters, random_state=random_state)
            algorithms[i].results['gap'] = gap
            algorithms[i].results['gap_sk'] = sk
            algorithms[i].results['gap_nc'] = nc
            algorithms[i].results['min_nc'] = min_clusters
            algorithms[i].results['max_nc'] = max_clusters

            sil, sil_nc, dav, dav_nc, cal, cal_nc, Z = Clustering.calc_scores(
                algorithms[i].estimator, X, min_clusters, max_clusters, random_state=random_state)
            algorithms[i].results['silhouette'] = sil
            algorithms[i].results['silhouette_nc'] = sil_nc
            algorithms[i].results['davies'] = dav
            algorithms[i].results['davies_nc'] = dav_nc
            algorithms[i].results['calinski'] = cal
            algorithms[i].results['calinski_nc'] = cal_nc
            algorithms[i].results['linkage'] = Z

        # Estimate number of clusters
        votes = np.zeros(algorithms[0].results['max_nc'] + 1)
        for alg in algorithms:
            votes[alg.results['gap_nc']] += 1
            votes[alg.results['silhouette_nc']] += 1
            votes[alg.results['davies_nc']] += 1
            votes[alg.results['calinski_nc']] += 1
        nc = list(np.where(votes == votes.max())[0])
        for alg in algorithms:
            alg.results['est_nc'] = nc
            alg.estimator.n_clusters = nc[0]
            alg.estimator = alg.estimator.fit(X)

    @classmethod
    def predict(cls, algorithms, data):
        ret = None
        for alg in algorithms:
            if hasattr(alg.estimator, 'predict'):
                if ret is None:
                    ret = {}
                ret[alg.name] = alg.estimator.predict(data)
        return ret
    
    @staticmethod
    def calc_inertia(data, labels):
        inertia = 0
        for nc in np.unique(labels):
            x = data[labels == nc, :]
            N = x.shape[0]
            s = 0
            for i1 in range(N - 1):
                for i2 in range(i1, N):
                    if i1 == i2: continue
                    s += np.sum((x[i1] - x[i2]) ** 2)
            inertia += s / N
        return inertia
    
    @classmethod
    def calc_gap(cls, model, data, min_clusters, max_clusters, num_iter=50, random_state=0, svd=True):
        if svd:
            U, s, V = np.linalg.svd(data, full_matrices=True)
            X = np.dot(data, V)
        else:
            X = data
        X_max = X.max(axis=0)
        X_min = X.min(axis=0)
        
        gap = []
        sk = []
        gap_star = []
        sk_star = []
        for nc in range(min_clusters, max_clusters+1):
            model.n_clusters = nc
            model.random_state = random_state
            pred_labels = model.fit_predict(data)
            if hasattr(model, 'inertia_'):
                dispersion = model.inertia_
            else:
                dispersion = Clustering.calc_inertia(data, pred_labels)

            ref_dispersions = []
            for iter in range(num_iter):
                np.random.seed(random_state + iter)
                ref = np.random.rand(*data.shape)
                ref = (ref * (X_max - X_min)) + X_min
                if svd:
                    ref = np.dot(ref, V)
                model.n_clusters = nc
                pred_labels = model.fit_predict(ref)
                if hasattr(model, 'inertia_'):
                    ref_dispersions.append(model.inertia_)
                else:
                    ref_dispersions.append(Clustering.calc_inertia(ref, pred_labels))
            ref_log_dispersion = np.mean(np.log(ref_dispersions))
            log_dispersion = np.log(dispersion)
            gap.append(ref_log_dispersion - log_dispersion)
            sd = np.std(np.log(ref_dispersions), ddof=0)
            sk.append(np.sqrt(1 + 1 /num_iter) * sd)

            gap_star.append(np.mean(ref_dispersions) - dispersion)
            sdk_star = np.std(ref_dispersions)
            sk_star.append(np.sqrt(1.0 + 1.0 / num_iter) * sdk_star)
        
        nc = -1
        for i in range(len(gap) - 1):
            if gap[i] >= gap[i + 1] - sk[i + 1]:
                nc = min_clusters + i
                break
        if nc == -1:
            nc = max_clusters

        return gap, sk, nc
    
    @classmethod
    def plot_gap(cls, algorithm, dname):
        if dname is None:
            return
        if not os.path.exists(dname):
            os.mkdir(dname)

        plt.figure()
        plt.title(algorithm.estimator.__class__.__name__)
        plt.xlabel("Number of clusters")
        plt.ylabel("Gap statistic")

        plt.plot(range(algorithm.results['min_nc'], algorithm.results['max_nc'] + 1),
                    algorithm.results['gap'], 'o-', color='dodgerblue')
        plt.errorbar(range(algorithm.results['min_nc'], algorithm.results['max_nc'] + 1),
                        algorithm.results['gap'], algorithm.results['gap_sk'], capsize=3)
        plt.axvline(x=algorithm.results['gap_nc'], ls='--', color='gray', zorder=0)
        plt.savefig('%s/gap_%s.png' %
                    (dname, algorithm.estimator.__class__.__name__),
                    bbox_inches='tight', dpi=75)
        plt.close()
    
    @classmethod
    def calc_scores(cls, model, data, min_clusters, max_clusters, random_state=0):
        silhouettes = []
        davieses = []
        calinskies = []
        if model.__class__.__name__ == 'HierarchicalClustering':
            linkage_matrix = model.fit(data)
        else:
            linkage_matrix = None
        for nc in range(min_clusters, max_clusters + 1):
            model.n_clusters = nc
            model.random_state = random_state
            pred_labels = model.fit_predict(data)
            silhouettes.append(silhouette_score(data, pred_labels, random_state=random_state))
            davieses.append(davies_bouldin_score(data, pred_labels))
            calinskies.append(calinski_harabasz_score(data, pred_labels))

        sil_nc = np.argmax(silhouettes) + min_clusters
        dav_nc = np.argmin(davieses) + min_clusters
        cal_nc = np.argmax(calinskies) + min_clusters

        return silhouettes, sil_nc, davieses, dav_nc, calinskies, cal_nc, linkage_matrix
    
    @classmethod
    def plot_silhouette(cls, algorithm, dname):
        if dname is None:
            return
        if not os.path.exists(dname):
            os.mkdir(dname)

        plt.figure()
        plt.title(algorithm.estimator.__class__.__name__)
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")

        plt.plot(range(algorithm.results['min_nc'], algorithm.results['max_nc'] + 1),
                    algorithm.results['silhouette'], 'o-', color='darkorange')
        plt.axvline(x=algorithm.results['silhouette_nc'], ls='--', color='gray', zorder=0)
        plt.savefig('%s/silhouette_%s.png' %
                    (dname, algorithm.estimator.__class__.__name__),
                    bbox_inches='tight', dpi=75)
        plt.close()

    @classmethod
    def plot_davies(cls, algorithm, dname):
        if dname is None:
            return
        if not os.path.exists(dname):
            os.mkdir(dname)

        plt.figure()
        plt.title(algorithm.estimator.__class__.__name__)
        plt.xlabel("Number of clusters")
        plt.ylabel("Davies-Bouldin score")

        plt.plot(range(algorithm.results['min_nc'], algorithm.results['max_nc'] + 1),
                    algorithm.results['davies'], 'o-', color='limegreen')
        plt.axvline(x=algorithm.results['davies_nc'], ls='--', color='gray', zorder=0)
        plt.savefig('%s/davies_%s.png' %
                    (dname, algorithm.estimator.__class__.__name__),
                    bbox_inches='tight', dpi=75)
        plt.close()

    @classmethod
    def plot_calinski(cls, algorithm, dname):
        if dname is None:
            return
        if not os.path.exists(dname):
            os.mkdir(dname)

        plt.figure()
        plt.title(algorithm.estimator.__class__.__name__)
        plt.xlabel("Number of clusters")
        plt.ylabel("Calinski and Harabasz score")

        plt.plot(range(algorithm.results['min_nc'], algorithm.results['max_nc'] + 1),
                    algorithm.results['calinski'], 'o-', color='crimson')
        plt.axvline(x=algorithm.results['calinski_nc'], ls='--', color='gray', zorder=0)
        plt.savefig('%s/calinski_%s.png' %
                    (dname, algorithm.estimator.__class__.__name__),
                    bbox_inches='tight', dpi=75)
        plt.close()

    @classmethod
    def plot_dendrogram(cls, algorithm, dname):
        if dname is None:
            return
        if not os.path.exists(dname):
            os.mkdir(dname)

        plt.figure()
        plt.title(algorithm.estimator.__class__.__name__)
        plt.title('Dendrogram')
        plt.ylabel('Distance')
        plt.xlabel(('A number in brackets represents the number of samples in the nodes\n'
                    'and a number not in brackets represents the index of the sample in the data.'))

        algorithm.estimator.dendrogram()
        plt.savefig('%s/dendrogram_%s.png' %
                    (dname, algorithm.estimator.__class__.__name__),
                    bbox_inches='tight', dpi=100)
        plt.close()

    @classmethod
    def make_report(cls, algorithms, data, dname, lang):
        if not os.path.exists(dname):
            os.mkdir(dname)
        
        shutil.copy(os.path.abspath(os.path.dirname(__file__)) + '/static/kmeans_mouse.png',
                    dname + '/kmeans_mouse.png')
        
        nc = ', '.join(map(str, algorithms[0].results['est_nc']))

        for algorithm in algorithms:
            Clustering.plot_gap(algorithm, dname)
            Clustering.plot_silhouette(algorithm, dname)
            Clustering.plot_davies(algorithm, dname)
            Clustering.plot_calinski(algorithm, dname)
            if algorithm.estimator.__class__.__name__ == 'HierarchicalClustering':
                Clustering.plot_dendrogram(algorithm, dname)

        env = Environment(
            loader=FileSystemLoader(
                os.path.abspath(
                    os.path.dirname(__file__)) + '/template', encoding='utf8'))
        if lang == 'jp':
            tmpl = env.get_template('report_clustering_jp.html.tmp')
        else:
            tmpl = env.get_template('report_clustering.html.tmp')

        html = tmpl.render(algorithms=algorithms, data=data, nc=nc).encode('utf-8')
        fo = io.open(dname + '/report.html', 'w', encoding='utf-8')
        fo.write(html.decode('utf-8'))
        fo.close()