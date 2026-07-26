#!/usr/bin/env python3
"""Perform K-means clustering using scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        k: The number of clusters.

    Returns:
        C: A numpy.ndarray of shape (k, d) containing the centroids.
        clss: A numpy.ndarray of shape (n,) containing the cluster
            index assigned to each data point.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
