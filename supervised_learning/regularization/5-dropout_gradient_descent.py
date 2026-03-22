#!/usr/bin/env python3
"""Gradient descent with Dropout regularization"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network using dropout GD"""
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)].copy()

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(i)] = weights["W{}".format(i)] - alpha * dW
        weights["b{}".format(i)] = weights["b{}".format(i)] - alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ)
            dZ *= (1 - cache["A{}".format(i - 1)] ** 2)
            dZ *= cache["D{}".format(i - 1)]
            dZ /= keep_prob
