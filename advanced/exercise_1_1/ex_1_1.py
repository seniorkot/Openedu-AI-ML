import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def print_results(file_name: str, variance_level: float):
    data = genfromtxt(file_name, delimiter=',')

    pca = PCA(n_components=2, svd_solver='full')
    x_transformed = pca.fit(data).transform(data)

    print('First coord: ' + str(x_transformed[0][0]))

    print('Second coord: ' + str(x_transformed[0][1]))

    explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_), 3)

    print('Variance 2: ' + str(explained_variance[1]))

    pca_2 = PCA(n_components=10, svd_solver='full')
    _ = pca_2.fit(data).transform(data)
    explained_variance_2 = \
        np.round(np.cumsum(pca_2.explained_variance_ratio_), 3)

    res = -1
    for i in range(10):
        if explained_variance_2[i] > variance_level:
            res = i + 1
            break

    print('Coords minimum for ' + str(variance_level) + ': ' + str(res))

    plt.scatter(x_transformed[:, 0], x_transformed[:, 1],
                edgecolor='none', s=40, cmap='winter')
    plt.show()
