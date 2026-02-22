#!/usr/bin/env python3
"""
Creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
    labels (numpy.ndarray): one-hot array of shape (m, classes)
                            containing correct labels
    logits (numpy.ndarray): one-hot array of shape (m, classes)
                            containing predicted labels

    Returns:
    numpy.ndarray: confusion matrix of shape (classes, classes)
                   rows = correct labels
                   columns = predicted labels
    """

    # Number of classes
    classes = labels.shape[1]

    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))

    # Convert one-hot to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Populate confusion matrix
    for t, p in zip(true_labels, predicted_labels):
        confusion[t][p] += 1

    return confusion
