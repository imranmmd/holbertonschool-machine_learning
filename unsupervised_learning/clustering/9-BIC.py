#!/usr/bin/env python3
"""Selects the best number of GMM clusters using the BIC."""

import numpy as np

expectation_maximization = __import__(
    '8-EM'
).expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Find the best number of GMM clusters using the BIC.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data set.
        kmin: A positive integer containing the minimum number of
            clusters to test.
        kmax: A positive integer containing the maximum number of
            clusters to test.
        iterations: A positive integer containing the maximum number
            of EM iterations.
        tol: A non-negative float containing the EM tolerance.
        verbose: A boolean determining whether EM output is printed.

    Returns:
        best_k: The number of clusters with the lowest BIC.
        best_result: A tuple containing pi, m, and S for best_k.
        l: A numpy.ndarray containing log likelihoods.
        b: A numpy.ndarray containing BIC values.

        On failure, returns None, None, None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None

    if kmin > kmax or kmax > n:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    cluster_numbers = np.arange(kmin, kmax + 1)
    likelihoods = np.zeros(cluster_numbers.shape[0])
    bic_values = np.zeros(cluster_numbers.shape[0])
    results = []

    for index, k in enumerate(cluster_numbers):
        pi, m, S, g, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        likelihoods[index] = likelihood

        parameters = (
            (k - 1)
            + (k * d)
            + (k * d * (d + 1) / 2)
        )

        bic_values[index] = (
            parameters * np.log(n) - 2 * likelihood
        )

    best_index = np.argmin(bic_values)
    best_k = cluster_numbers[best_index]
    best_result = results[best_index]

    return best_k, best_result, likelihoods, bic_values
