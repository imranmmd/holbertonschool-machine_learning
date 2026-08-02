#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
            Dataset whose dimensions have mean 0.
        var: fraction of variance to maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd)
            Weight matrix for the transformed data.
    """
    # SVD of the centered data
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    # Variance associated with each principal component
    variance = S ** 2

    # Cumulative fraction of variance
    cumulative = np.cumsum(variance) / np.sum(variance)

    # Number of components needed to maintain `var`
    nd = np.argmax(cumulative >= var) + 1

    # Vh contains principal components as rows
    # Transpose to obtain shape (d, nd)
    W = Vh[:nd].T

    return W
