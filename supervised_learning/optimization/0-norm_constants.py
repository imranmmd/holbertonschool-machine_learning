#!/usr/bin/env python3
"""
Calculates the normalization constants (mean and std) for a dataset.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation for each feature in X.

    Args:
        X (numpy.ndarray): shape (m, nx) where
            m = number of data points
            nx = number of features

    Returns:
        tuple: (mean, std) of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)  # population std
    return mean, std
