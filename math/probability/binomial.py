#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):

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

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = mean / p

            self.n = round(n)
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the PMF for k successes"""
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0 or k > self.n:
            return 0

        numerator = 1
        denominator = 1

        for i in range(1, k + 1):
            numerator *= (self.n - i + 1)
            denominator *= i

        comb = numerator / denominator

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the CDF for k successes"""
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        if k > self.n:
            k = self.n

        cumulative = 0
        for i in range(0, k + 1):
            cumulative += self.pmf(i)

        return cumulative
