#!/usr/bin/env python3
"""Bayesian Optimization module"""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Args:
            f: black-box function to optimize
            X_init: initial sample inputs
            Y_init: initial sample outputs
            bounds: tuple (min, max)
            ac_samples: number of acquisition samples
            l: kernel length parameter
            sigma_f: kernel output standard deviation
            xsi: exploration-exploitation factor
            minimize: True for minimization, False for maximization
        """
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
