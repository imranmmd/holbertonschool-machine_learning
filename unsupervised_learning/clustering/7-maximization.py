#!/usr/bin/env python3
"""Maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """Calculate the maximization step of the EM algorithm for a GMM.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data set.
        g: A numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.

    Returns:
        pi: A numpy.ndarray of shape (k,) containing the updated priors.
        m: A numpy.ndarray of shape (k, d) containing the updated means.
        S: A numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices.

        On failure, returns None, None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, points = g.shape

    if points != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    totals = np.sum(g, axis=1)

    pi = totals / n
    m = np.matmul(g, X) / totals[:, np.newaxis]
    S = np.zeros((k, d, d))

    for cluster in range(k):
        difference = X - m[cluster]
        weighted = difference * g[cluster, :, np.newaxis]
        S[cluster] = np.matmul(weighted.T, difference) / totals[cluster]

    return pi, m, S
