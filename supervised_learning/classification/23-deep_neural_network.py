#!/usr/bin/env python3
"""Module that defines a deep neural network."""

import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        layers_arr = np.array(layers)
        if (len(layers_arr.shape) != 1 or
                not np.issubdtype(layers_arr.dtype, np.integer) or
                np.any(layers_arr <= 0)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network."""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            w = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            a_prev = self.__cache["A{}".format(i - 1)]
            z = np.matmul(w, a_prev) + b
            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-z))

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the cost using logistic regression."""
        m = Y.shape[1]
        return -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the network."""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dz = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            a_prev = cache["A{}".format(i - 1)]
            w = weights_copy["W{}".format(i)]

            dw = np.matmul(dz, a_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.__weights["W{}".format(i)] = (
                self.__weights["W{}".format(i)] - alpha * dw
            )
            self.__weights["b{}".format(i)] = (
                self.__weights["b{}".format(i)] - alpha * db
            )

            if i > 1:
                a = cache["A{}".format(i - 1)]
                dz = np.matmul(w.T, dz) * a * (1 - a)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        iters = []
        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    iters.append(i)
                    costs.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
