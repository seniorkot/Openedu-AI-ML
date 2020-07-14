import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


def print_results(scores_fn: str, loadings_fn: str):
    scores = genfromtxt(scores_fn, delimiter=';')
    loadings = genfromtxt(loadings_fn, delimiter=';')

    values = np.dot(scores, loadings.T)

    plt.imshow(values, cmap='Greys_r')
    plt.show()
