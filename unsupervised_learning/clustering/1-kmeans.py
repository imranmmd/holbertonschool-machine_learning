#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering

    Args:
        X: dataset (n, d)
        k: number of clusters
        iterations: max iterations

    Returns:
        C: centroids (k, d)
        clss: cluster labels (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if k > X.shape[0]:
        return None, None

    # initialize centroids (1st allowed random call inside initialize)
    C = initialize(X, k)

    for _ in range(iterations):

        # Assignment step
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        new_C = np.zeros_like(C)

        # Update step
        for i in range(k):  # allowed loop #1
            points = X[clss == i]

            if points.shape[0] == 0:
                # reinitialize empty cluster (2nd allowed random call)
                new_C[i] = initialize(X, 1)
            else:
                new_C[i] = points.mean(axis=0)

        # Convergence check
        if np.allclose(C, new_C):
            return C, clss

        C = new_C

    return C, clss
