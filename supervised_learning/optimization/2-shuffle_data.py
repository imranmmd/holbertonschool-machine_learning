#!/usr/bin/env python3
"""
Shuffles two matrices (X and Y) in unison.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles data points in X and Y the same way.

    Args:
        X (numpy.ndarray): shape (m, nx), first dataset
        Y (numpy.ndarray): shape (m, ny), second dataset

    Returns:
        tuple: shuffled X and Y
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
