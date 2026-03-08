#!/usr/bin/env python3
"""Module that defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector."""
        return self.__W

    @property
    def b(self):
        """Get the bias."""
        return self.__b

    @property
    def A(self):
        """Get the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost using logistic regression."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron."""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T).T / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
