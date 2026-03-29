#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix using the given mean and std.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes the input data X.

    Args:
        X (numpy.ndarray): shape (d, nx), data to normalize
        m (numpy.ndarray): shape (nx,), mean of each feature
        s (numpy.ndarray): shape (nx,), standard deviation of each feature

    Returns:
        numpy.ndarray: normalized X
    """
    return (X - m) / s
