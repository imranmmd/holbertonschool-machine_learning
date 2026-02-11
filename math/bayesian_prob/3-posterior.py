#!/usr/bin/env python3
"""Module for calculating posterior probability"""
import numpy as np


def posterior(x, n, P, Pr):
    """Calculates the posterior probability of each probability in P
    given x successes in n trials and prior Pr.

    Args:
        x (int): number of successes
        n (int): number of trials
        P (np.ndarray): 1D array of hypothetical probabilities
        Pr (np.ndarray): 1D array of prior probabilities (same shape as P)

    Returns:
        np.ndarray: posterior probabilities for each p in P

    Raises:
        ValueError or TypeError with messages as specified in the assignment.
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

    # 4. P must be a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 5. Pr must be a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # 6. All values in P and Pr must be in [0, 1] (check P first, then Pr)
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # 7. Pr must sum to 1
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    # Compute log(C(n, x)) using sums of logs (only numpy used)
    # log(k!) = sum_{i=1..k} log(i); handle k==0 by using empty sum -> 0
    def log_factorial(k):
        if k <= 1:
            return 0.0
        return np.sum(np.log(np.arange(1, k + 1, dtype=float)))

    log_comb = log_factorial(n) - log_factorial(x) - log_factorial(n - x)

    # Compute log-likelihood terms safely (handle p==0 or p==1 and x==0 or n-x==0)
    if x == 0:
        log_p_term = np.zeros_like(P, dtype=float)
    else:
        # x * log(P); if P == 0 and x > 0, set to -inf
        log_p = np.log(P)
        log_p_term = x * log_p
        log_p_term = np.where(P == 0, -np.inf, log_p_term)

    nx = n - x
    if nx == 0:
        log_1mp_term = np.zeros_like(P, dtype=float)
    else:
        # (n-x) * log(1-P); if P == 1 and n-x > 0, set to -inf
        log_1mp = np.log(1 - P)
        log_1mp_term = nx * log_1mp
        log_1mp_term = np.where(P == 1, -np.inf, log_1mp_term)

    log_likelihood = log_comb + log_p_term + log_1mp_term

    # Convert back to likelihood (may underflow for very small values)
    likelihood = np.exp(log_likelihood)

    # intersections = likelihood * prior
    intersections = likelihood * Pr

    # marginal likelihood
    marginal = np.sum(intersections)

    # posterior: intersections / marginal (will be numpy array)
    posterior = intersections / marginal

    return posterior
