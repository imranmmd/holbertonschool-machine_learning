#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if k > X.shape[0]:
        return None, None

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)

    C = np.random.uniform(xmin, xmax, size=(k, X.shape[1]))

    for _ in range(iterations):
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        C_prev = C.copy()

        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                C[i] = np.random.uniform(xmin, xmax, size=(1, X.shape[1]))
            else:
                C[i] = points.mean(axis=0)

        if np.array_equal(C, C_prev):
            dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(dist, axis=1)
            return C, clss

    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(dist, axis=1)

    return C, clss
