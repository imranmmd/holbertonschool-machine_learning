#!/usr/bin/env python3
"""Finds the optimal number of clusters for K-means"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Returns:
        results, d_vars or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmin > kmax:
        return None, None
    if kmax > X.shape[0]:
        return None, None
    if (kmax - kmin + 1) < 2:
        return None, None

    results = []
    vars_list = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        vars_list.append(variance(X, C))

    base_var = vars_list[0]
    d_vars = np.array(vars_list) - base_var

    return results, d_vars
