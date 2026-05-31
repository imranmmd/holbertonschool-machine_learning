#!/usr/bin/env python3
"""Bayesian Optimization module"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D GP"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f

        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        sigma = np.where(sigma == 0, 1e-9, sigma)

        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # STRICT duplicate check (this is what fixes your bug)
            if np.any(np.isclose(self.gp.X.flatten(), X_next[0])):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
