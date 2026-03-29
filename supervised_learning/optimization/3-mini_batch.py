#!/usr/bin/env python3
"""
Create mini-batches for gradient descent
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Splits the dataset (X, Y) into mini-batches.

    Args:
        X (numpy.ndarray): shape (m, nx), input data
        Y (numpy.ndarray): shape (m, ny), labels
        batch_size (int): number of examples per mini-batch

    Returns:
        list of tuples: [(X_batch, Y_batch), ...]
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
