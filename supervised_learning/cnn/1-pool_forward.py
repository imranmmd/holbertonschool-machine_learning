#!/usr/bin/env python3
"""Pooling forward propagation module"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer

    Parameters:
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'

    Returns:
    The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    # Initialize output
    A = np.zeros((m, h_new, w_new, c_prev))

    # Perform pooling
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice_prev = A_prev[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(slice_prev)
                    else:
                        A[i, h, w, c] = np.mean(slice_prev)

    return A
