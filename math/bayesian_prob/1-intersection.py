#!/usr/bin/env python3
"""Module for calculating the intersection of Bayesian probabilities"""

import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining x successes in n trials
    with each hypothetical probability in P"""

    # 1. Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2. Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # 3. x cannot be greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 4. Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5. Validate Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    # 6. Check values of P
    if np.any((P < 0) | (P > 1)):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    # 7. Check values of Pr
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError(
            "All values in Pr must be in the range [0, 1]"
        )

    # 8. Check Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute binomial coefficient
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)

    coeff = n_fact / (x_fact * nx_fact)

    # Likelihood for each P
    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = likelihood * prior
    return likelihood * Pr
