#!/usr/bin/env python3
"""Find the optimal number of GMM clusters using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Find the best number of GMM clusters using BIC.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data set.
        kmin: A positive integer containing the minimum number of
            clusters to test.
        kmax: A positive integer containing the maximum number of
            clusters to test.
        iterations: A positive integer containing the maximum number
            of EM iterations.
        tol: A non-negative float containing the EM tolerance.
        verbose: A boolean determining whether EM information should
            be printed.

    Returns:
        best_k: The number of clusters with the lowest BIC.
        best_result: A tuple containing pi, m, and S for best_k.
        l: A numpy.ndarray containing the log likelihoods.
        b: A numpy.ndarray containing the BIC values.

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

    if kmax < kmin:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    count = kmax - kmin + 1
    log_likelihoods = np.zeros(count)
    bic_values = np.zeros(count)
    results = []

    for index, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        log_likelihoods[index] = likelihood

        parameters = (
            k * d
            + k * d * (d + 1) / 2
            + k - 1
        )

        bic_values[index] = (
            parameters * np.log(n) - 2 * likelihood
        )

    best_index = np.argmin(bic_values)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log_likelihoods, bic_values
