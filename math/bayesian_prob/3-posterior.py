#!/usr/bin/env python3
"""Module for calculating posterior probability"""


def posterior(x, n, P, Pr):
    """Calculates the posterior probability"""

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # Check x > n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not hasattr(P, "ndim") or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate Pr
    if (not hasattr(Pr, "shape") or
            not hasattr(P, "shape") or
            Pr.shape != P.shape):
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    # Check values in P
    for p in P:
        if p < 0 or p > 1:
            raise ValueError(
                "All values in P must be in the range [0, 1]"
            )

    # Check values in Pr
    for pr in Pr:
        if pr < 0 or pr > 1:
            raise ValueError(
                "All values in Pr must be in the range [0, 1]"
            )

    # Check sum of Pr
    if abs(sum(Pr) - 1) > 1e-8:
        raise ValueError("Pr must sum to 1")

    # Factorial helper
    def factorial(num):
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result

    # Combination C(n, x)
    comb = factorial(n) / (factorial(x) * factorial(n - x))

    # Compute intersections
    intersections = []
    for i in range(len(P)):
        likelihood = comb * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        intersections.append(likelihood * Pr[i])

    # Compute marginal
    marginal = sum(intersections)

    # Compute posterior
    post = []
    for value in intersections:
        post.append(value / marginal)

    # Return same type as P
    return type(P)(post)
