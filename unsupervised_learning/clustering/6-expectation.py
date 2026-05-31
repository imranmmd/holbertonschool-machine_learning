#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf

def expectation(X, pi, m, S):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (not isinstance(pi, np.ndarray) or pi.shape != (k,) or
        not isinstance(m, np.ndarray) or m.shape != (k, d) or
        not isinstance(S, np.ndarray) or S.shape != (k, d, d)):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    sum_g = np.sum(g, axis=0, keepdims=True)
    g = g / sum_g

    log_likelihood = np.sum(np.log(sum_g))

    return g, log_likelihood
