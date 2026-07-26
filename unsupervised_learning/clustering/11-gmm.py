#!/usr/bin/env python3
"""Calculate a Gaussian Mixture Model using scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Calculate a Gaussian Mixture Model from a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        k: The number of Gaussian clusters.

    Returns:
        pi: A numpy.ndarray of shape (k,) containing cluster priors.
        m: A numpy.ndarray of shape (k, d) containing cluster means.
        S: A numpy.ndarray of shape (k, d, d) containing covariance
            matrices.
        clss: A numpy.ndarray of shape (n,) containing the cluster
            index assigned to each data point.
        bic: The Bayesian Information Criterion value of the model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
