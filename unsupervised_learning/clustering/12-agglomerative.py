#!/usr/bin/env python3
"""Perform agglomerative clustering using Ward linkage."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        dist: The maximum cophenetic distance between clusters.

    Returns:
        A numpy.ndarray of shape (n,) containing the cluster index
        assigned to each data point.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    plt.show()

    return clss
