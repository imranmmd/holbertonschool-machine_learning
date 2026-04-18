#!/usr/bin/env python3
"""Pooling backward propagation module"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer

    Parameters:
    dA: numpy.ndarray of shape (m, h_new, w_new, c)
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'

    Returns:
    dA_prev
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        slice_prev = A_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ]

                        mask = (slice_prev == np.max(slice_prev))

                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ] += mask * dA[i, h, w, ch]

                    else:  # avg pooling
                        da = dA[i, h, w, ch] / (kh * kw)

                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ] += np.ones((kh, kw)) * da

    return dA_prev
