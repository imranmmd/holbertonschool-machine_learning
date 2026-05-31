#!/usr/bin/env python3
"""K-means initialization function"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Args:
        X: numpy.ndarray of shape (n, d)
        k: number of clusters

    Returns:
        numpy.ndarray of shape (k, d) or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if k > X.shape[0]:
        return None

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)

    return np.random.uniform(xmin, xmax, size=(k, X.shape[1]))
