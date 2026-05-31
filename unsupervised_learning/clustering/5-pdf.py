#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    X: (n, d)
    m: (d,)
    S: (d, d)
    Returns: (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None

    d = X.shape[1]

    # determinant and inverse
    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    diff = X - m
    exp_term = np.sum(diff @ inv * diff, axis=1)

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    P = (1.0 / denom) * np.exp(-0.5 * exp_term)

    # numerical stability constraint
    P = np.maximum(P, 1e-300)

    return P
