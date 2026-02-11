#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution
        """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Mean
            mean = sum(data) / len(data)

            # Variance (population)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Solve for p
            p = 1 - (variance / mean)

            # Solve for n
            n = mean / p

            # Round n
            self.n = round(n)

            # Recalculate p using rounded n
            self.p = mean / self.n
