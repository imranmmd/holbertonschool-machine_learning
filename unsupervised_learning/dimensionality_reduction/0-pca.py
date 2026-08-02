#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
            Dataset with mean 0 in every dimension.
        var: fraction of the variance that PCA transformation should
            maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd)
            Weight matrix that maintains `var` fraction of variance.
    """
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    variance = S ** 2
    cumulative = np.cumsum(variance)
    cumulative /= np.sum(variance)

    nd = np.argmax(cumulative >= var) + 1

    return Vh[:nd].T
