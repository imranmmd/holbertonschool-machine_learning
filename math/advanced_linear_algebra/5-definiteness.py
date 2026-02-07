#!/usr/bin/env python3
"""
Matrix definiteness
"""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""

    # Type check
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Must be non-empty square matrix
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.size == 0
    ):
        return None

    # Must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)

    # Classify definiteness
    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
