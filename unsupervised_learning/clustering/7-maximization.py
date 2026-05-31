#!/usr/bin/env python3
"""Maximization step for a GMM"""

import numpy as np


def maximization(X, g):
    """Performs the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n2 = g.shape

    if n2 != n:
        return None, None, None

    Nk = np.sum(g, axis=1)  # (k,)

    if np.any(Nk == 0):
        return None, None, None

    # priors
    pi = Nk / n

    # means
    m = np.zeros((k, d))
    for i in range(k):
        m[i] = np.sum(g[i, :, None] * X, axis=0) / Nk[i]

    # covariances
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i, :, None] * diff).T @ diff / Nk[i]

    return pi, m, S
