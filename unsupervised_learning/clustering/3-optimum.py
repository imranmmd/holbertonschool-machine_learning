#!/usr/bin/env python3
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if k > X.shape[0]:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        old_C = C.copy()

        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        for i in range(k):
            pts = X[clss == i]
            if pts.shape[0] == 0:
                C[i] = initialize(X, 1)  # safe allowed reuse
            else:
                C[i] = pts.mean(axis=0)

        if np.allclose(old_C, C):
            break

    return C, clss
