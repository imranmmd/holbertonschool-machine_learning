#!/usr/bin/env python3
"""Module for calculating posterior probability"""


def posterior(x, n, P, Pr):
    """Calculates the posterior probability of each probability in P
    given x successes in n trials and prior Pr.

    Args:
        x (int): number of successes
        n (int): number of trials
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: numpy.ndarray with prior probabilities (same shape as P)

    Returns:
        numpy.ndarray: posterior probabilities for each p in P

    Raises:
        ValueError or TypeError with exact messages required by the task.
    """
    # 1. n must be a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2. x must be an integer >= 0
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # 3. x cannot be greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 4. P must be a 1D numpy.ndarray (duck-check without importing numpy)
    if not hasattr(P, "ndim") or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5. Pr must be a numpy.ndarray with the same shape as P
    if not hasattr(Pr, "shape") or not hasattr(P, "shape") or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # 6. All values in P and Pr must be in [0, 1] (check P first, then Pr)
    for val in P:
        if val < 0 or val > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for val in Pr:
        if val < 0 or val > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")

    # 7. Pr must sum to 1
    if abs(sum(Pr) - 1.0) > 1e-8:
        raise ValueError("Pr must sum to 1")

    # Compute binomial coefficient C(n, x) using multiplicative formula
    # This returns a float and avoids huge intermediate factorials
    comb = 1.0
    # If x is 0 the loop is skipped and comb stays 1.0
    for i in range(1, x + 1):
        comb *= (n - x + i) / i

    # Compute likelihoods and intersections
    intersections = []
    nx = n - x
    for i, p in enumerate(P):
        # p**x and (1-p)**(n-x) handle edge cases (0**0 == 1 in Python)
        try:
            likelihood = comb * (p ** x) * ((1 - p) ** nx)
        except Exception:
            # In case numeric type from P raises unexpected error, coerce to float
            pv = float(p)
            likelihood = comb * (pv ** x) * ((1 - pv) ** nx)
        intersections.append(likelihood * Pr[i])

    # Marginal likelihood
    marginal = sum(intersections)

    # Posterior = intersections / marginal
    posterior_list = [val / marginal for val in intersections]

    # Return same type as P (NumPy ndarray) by leveraging P * 0 + posterior_list.
    # If P is a numpy.ndarray this yields an ndarray; tests expect that behavior.
    return P * 0 + posterior_list
