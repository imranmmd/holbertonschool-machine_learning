#!/usr/bin/env python3
"""One-hot encoding module."""

import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix."""
    if (not isinstance(Y, np.ndarray) or len(Y.shape) != 1 or
            not isinstance(classes, int) or classes <= 0):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
