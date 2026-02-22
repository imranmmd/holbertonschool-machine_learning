#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class

    Parameters:
    confusion (numpy.ndarray): shape (classes, classes)
        rows -> actual labels
        columns -> predicted labels

    Returns:
    numpy.ndarray: shape (classes,)
        precision for each class
    """

    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # Total predicted positives per class = sum of each column
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    # which equals TP / column_sum
    precision = true_positives / predicted_positives

    return precision
