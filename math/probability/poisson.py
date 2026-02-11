#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution

        Parameters:
        data (list): list of data to estimate the distribution
        lambtha (float): expected number of occurrences
        """

        if data is None:
            # Use provided lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Estimate lambtha as the mean of the data
            self.lambtha = float(sum(data) / len(data))
