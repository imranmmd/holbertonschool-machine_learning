#!/usr/bin/env python3
"""Module for calculating marginal probability"""


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining x successes in n trials"""

    # 1. n validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2. x validation
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # 3. x > n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 4. P validation (must behave like 1D numpy array)
    if not hasattr(P, "ndim") or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5. Pr validation
    if (not hasattr(Pr, "shape") or
            not hasattr(P, "shape") or
            Pr.shape != P.shape):
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    # 6. P range check
    for p in P:
        if p < 0 or p > 1:
            raise ValueError(
                "All values in P must be in the range [0, 1]"
            )

    # 7. Pr range check
    for pr in Pr:
        if pr < 0 or pr > 1:
            raise ValueError(
                "All values in Pr must be in the range [0, 1]"
            )

    # 8. Pr sum check
    if abs(sum(Pr) - 1) > 1e-8:
        raise ValueError("Pr must sum to 1")

    # ---- Compute likelihood manually ----

    # Compute combination C(n, x)
    def factorial(num):
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result

    comb = factorial(n) / (factorial(x) * factorial(n - x))

    marginal_prob = 0

    for i in range(len(P)):
        likelihood = comb * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        marginal_prob += likelihood * Pr[i]

    return marginal_prob
