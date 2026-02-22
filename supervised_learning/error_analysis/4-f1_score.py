#!/usr/bin/env python3
"""
Calculates the F1 score for each class in a confusion matrix
"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class

    Parameters:
    confusion (numpy.ndarray): shape (classes, classes)
        rows -> actual labels
        columns -> predicted labels

    Returns:
    numpy.ndarray: shape (classes,)
        F1 score for each class
    """

    # Get recall (sensitivity) and precision
    recall = sensitivity(confusion)
    prec = precision(confusion)

    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
