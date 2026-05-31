#!/usr/bin/env python3
"""Maximization step for a GMM"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n2 = g.shape

    if n2 != n:
        return None, None, None
    if np.any(g < 0) or np.any(g > 1):
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None

    pi = Nk / n
    m = (g @ X) / Nk[:, np.newaxis]

    diff = X[np.newaxis, :, :] - m[:, np.newaxis, :]
    S = np.einsum('kn,kni,knj->kij', g, diff, diff)
    S = S / Nk[:, np.newaxis, np.newaxis]

    return pi, m, S
