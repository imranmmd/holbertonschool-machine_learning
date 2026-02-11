#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for k successes

        P(X = k) = (e^(-λ) * λ^k) / k!
        """
        # Convert k to integer if possible
        try:
            k = int(k)
        except Exception:
            return 0

        # If k is negative → out of range
        if k < 0:
            return 0

        # Compute k!
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # e approximation
        e = 2.7182818285

        return (e ** (-self.lambtha) *
                (self.lambtha ** k) /
                factorial)
