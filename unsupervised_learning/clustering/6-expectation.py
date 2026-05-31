#!/usr/bin/env python3
"""Expectation step for a GMM using EM algorithm"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the E-step of the EM algorithm

    X: (n, d)
    pi: (k,)
    m: (k, d)
    S: (k, d, d)

    Returns:
        g: (k, n)
        l: log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    # responsibilities
    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    # log likelihood
    likelihood = np.sum(g, axis=0)
    l = np.sum(np.log(likelihood))

    # normalize responsibilities
    g = g / likelihood

    return g, l
