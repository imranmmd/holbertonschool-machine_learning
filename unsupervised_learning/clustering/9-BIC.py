#!/usr/bin/env python3
"""Expectation-Maximization algorithm for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform expectation-maximization for a GMM.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data set.
        k: A positive integer containing the number of clusters.
        iterations: A positive integer containing the maximum number
            of iterations.
        tol: A non-negative float containing the tolerance used for
            early stopping.
        verbose: A boolean determining whether information is printed.

    Returns:
        pi: A numpy.ndarray containing the cluster priors.
        m: A numpy.ndarray containing the cluster means.
        S: A numpy.ndarray containing the covariance matrices.
        g: A numpy.ndarray containing posterior probabilities.
        l: The log likelihood of the model.

        On failure, returns None, None, None, None, None.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None:
        return None, None, None, None, None

    g, likelihood = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(
            round(likelihood, 5)
        ))

    for iteration in range(1, iterations + 1):
        pi, m, S = maximization(X, g)

        if pi is None:
            return None, None, None, None, None

        g, new_likelihood = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        converged = abs(new_likelihood - likelihood) <= tol
        likelihood = new_likelihood

        if verbose and (
            iteration % 10 == 0
            or converged
            or iteration == iterations
        ):
            print("Log Likelihood after {} iterations: {}".format(
                iteration, round(likelihood, 5)
            ))

        if converged:
            break

    return pi, m, S, g, likelihood
