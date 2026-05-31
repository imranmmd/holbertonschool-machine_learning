#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds optimal k using variance elbow method
    """

    # -------- validation --------
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmin > kmax:
        return None, None

    # must analyze at least 2 k values
    if (kmax - kmin + 1) < 2:
        return None, None

    results = []
    vars_list = []

    base_var = None

    # -------- loop (ONLY loop #1 allowed) --------
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        v = variance(X, C, clss)

        if base_var is None:
            base_var = v

        results.append((C, clss))
        vars_list.append(v)

    # -------- compute deltas (NO extra loop allowed) --------
    d_vars = np.array(base_var - np.array(vars_list))

    return results, d_vars
