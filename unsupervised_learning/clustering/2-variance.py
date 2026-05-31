#!/usr/bin/env python3
"""Variance for K-means clustering"""

import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[0] == 0 or C.shape[0] == 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(dist, axis=1)

    closest = C[clss]

    return np.sum((X - closest) ** 2)
