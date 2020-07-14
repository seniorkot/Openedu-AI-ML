import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def print_results(data: pd.DataFrame, centroid: np.ndarray, cluster: int):
    coords = data.drop('Cluster', axis=1)

    kmeans = KMeans(n_clusters=3, init=centroid, max_iter=100, n_init=1)

    model = kmeans.fit(coords)
    print('Clusters: ' + str(model.labels_.tolist()))

    alldistances = kmeans.fit_transform(coords)

    labels = model.labels_.tolist()
    dist = [alldistances[i][cluster] for i in range(len(labels)) if
            labels[i] == cluster]
    print('Avg distance: ' + str(sum(dist) / len(dist)))
