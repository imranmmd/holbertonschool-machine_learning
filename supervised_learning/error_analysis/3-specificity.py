#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class

    Parameters:
    confusion (numpy.ndarray): shape (classes, classes)
        rows -> actual labels
        columns -> predicted labels

    Returns:
    numpy.ndarray: shape (classes,)
        specificity for each class
    """

    classes = confusion.shape[0]

    # True Positives
    TP = np.diag(confusion)

    # False Positives (column sum minus TP)
    FP = np.sum(confusion, axis=0) - TP

    # False Negatives (row sum minus TP)
    FN = np.sum(confusion, axis=1) - TP

    # Total samples
    total = np.sum(confusion)

    # True Negatives
    TN = total - (TP + FP + FN)

    # Specificity = TN / (TN + FP)
    specificity = TN / (TN + FP)

    return specificity
