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

    C = np.random.uniform(xmin, xmax, (k, X.shape[1]))

    for _ in range(iterations):  # LOOP 1

        # assignment step (vectorized)
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        new_C = np.zeros_like(C)

        for i in range(k):  # LOOP 2
            points = X[clss == i]

            if points.shape[0] == 0:
                new_C[i] = np.random.uniform(xmin, xmax)
            else:
                new_C[i] = points.mean(axis=0)

        if np.allclose(C, new_C):
            return C, clss

        C = new_C

    return C, clss
