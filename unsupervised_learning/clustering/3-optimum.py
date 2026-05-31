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

    # init centroids (uses 1st allowed uniform via initialize)
    C = initialize(X, k)

    for _ in range(iterations):  # loop #1
        old_C = C.copy()

        # Assignment step
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)   # FIX IS HERE

        # Update step
        for i in range(k):  # loop #2 (only loop allowed inside)
            pts = X[clss == i]
            if pts.shape[0] == 0:
                # second allowed random use (direct uniform)
                C[i] = np.random.uniform(X.min(axis=0), X.max(axis=0))
            else:
                C[i] = pts.mean(axis=0)

        # convergence check
        if np.allclose(old_C, C):
            break

    return C, clss
