#!/usr/bin/env python3
"""Gaussian Process module"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix
        using the RBF kernel
        """
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sqdist = sqdist + np.sum(X2 ** 2, axis=1)
        sqdist = sqdist - 2 * np.matmul(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(
            -0.5 * sqdist / (self.l ** 2)
        )

    def predict(self, X_s):
        """
        Predicts the mean and variance of points
        in a Gaussian process

        Args:
            X_s: numpy.ndarray of shape (s, 1)

        Returns:
            mu, sigma
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        sigma = np.diag(
            K_ss - K_s.T @ K_inv @ K_s
        )

        return mu, sigma
