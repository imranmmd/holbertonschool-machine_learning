#!/usr/bin/env python3
"""Bayesian Optimization module"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor"""
        self.f = f

        self.gp = GP(
            X_init,
            Y_init,
            l=l,
            sigma_f=sigma_f
        )

        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location

        Returns:
            X_next, EI
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            optimum = np.min(self.gp.Y)
            improvement = optimum - mu - self.xsi
        else:
            optimum = np.max(self.gp.Y)
            improvement = mu - optimum - self.xsi

        with np.errstate(divide='ignore'):
            Z = improvement / sigma

        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
