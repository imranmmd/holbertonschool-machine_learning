#!/usr/bin/env python3
"""Calculates the maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data.
        g: A numpy.ndarray of shape (k, n) containing posterior
            probabilities for each cluster.

    Returns:
        pi: A numpy.ndarray of shape (k,) containing updated priors.
        m: A numpy.ndarray of shape (k, d) containing updated means.
        S: A numpy.ndarray of shape (k, d, d) containing updated
            covariance matrices.

        Returns (None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, g_n = g.shape

    if n == 0 or d == 0 or k == 0 or g_n != n:
        return None, None, None

    if np.any(g < 0) or np.any(g > 1):
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    cluster_weights = np.sum(g, axis=1)

    if np.any(cluster_weights == 0):
        return None, None, None

    pi = cluster_weights / n
    m = np.matmul(g, X) / cluster_weights[:, np.newaxis]
    S = np.zeros((k, d, d))

    for cluster in range(k):
        difference = X - m[cluster]
        weighted_difference = difference * g[cluster, :, np.newaxis]
        S[cluster] = np.matmul(weighted_difference.T, difference)
        S[cluster] /= cluster_weights[cluster]

    return pi, m, S
