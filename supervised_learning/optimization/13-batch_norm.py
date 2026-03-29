#!/usr/bin/env python3
"""
Batch Normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Args:
        Z (np.ndarray): shape (m, n) — unactivated outputs
        gamma (np.ndarray): shape (1, n) — scaling parameters
        beta (np.ndarray): shape (1, n) — offset parameters
        epsilon (float): small number to avoid division by zero

    Returns:
        np.ndarray: normalized Z
    """
    # Compute mean and variance for each feature
    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    # Normalize
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)

    # Scale and shift
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
