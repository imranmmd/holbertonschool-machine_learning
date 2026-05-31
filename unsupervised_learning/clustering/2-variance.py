#!/usr/bin/env python3
"""Variance for K-means clustering"""

import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance

    Args:
        X: (n, d) data points
        C: (k, d) centroids

    Returns:
        total variance or None
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[0] == 0 or C.shape[0] == 0:
        return None

    # compute distance from each point to each centroid
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    # assign each point to closest centroid
    clss = np.argmin(dist, axis=1)

    # compute squared distances to assigned centroid
    closest = C[clss]

    var = np.sum((X - closest) ** 2)

    return var
