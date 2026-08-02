#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
           Dataset with mean 0 for each dimension.
        var: fraction of variance to maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd)
           Weight matrix containing the principal components.
    """
    # Compute covariance matrix
    C = np.cov(X, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort from largest eigenvalue to smallest
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate cumulative explained variance
    explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Smallest number of components maintaining `var`
    nd = np.searchsorted(explained_variance, var) + 1

    # Return corresponding eigenvectors
    return eigenvectors[:, :nd]
