#!/usr/bin/env python3
"""
Calculates the sensitivity (recall) for each class in a confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class

    Parameters:
    confusion (numpy.ndarray): shape (classes, classes)
        rows -> actual labels
        columns -> predicted labels

    Returns:
    numpy.ndarray: shape (classes,)
        sensitivity for each class
    """

    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # Total actual positives per class = sum of each row
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity (Recall) = TP / (TP + FN)
    # which equals TP / row_sum
    sensitivity = true_positives / actual_positives

    return sensitivity
