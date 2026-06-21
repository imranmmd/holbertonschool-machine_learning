#!/usr/bin/env python3
"""RNN Cell"""

import numpy as np


class RNNCell:
    """Represents a simple RNN cell"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: previous hidden state, shape (m, h)
            x_t: input data, shape (m, i)

        Returns:
            h_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
