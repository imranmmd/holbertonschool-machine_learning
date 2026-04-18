#!/usr/bin/env python3
"""Convolutional backward propagation module"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer

    Parameters:
    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b: numpy.ndarray of shape (1, 1, 1, c_new)
    padding: "same" or "valid"
    stride: tuple of (sh, sw)

    Returns:
    dA_prev, dW, db
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    # Padding
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = 0
        pw = 0

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)

    # Initialize gradients
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Backprop
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    slice_prev = A_prev_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]

                    dA_prev_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += (
                        slice_prev * dZ[i, h, w, c]
                    )

    # Remove padding from dA_prev
    if padding == "same":
        dA_prev = dA_prev_pad[
            :,
            ph:-ph if ph != 0 else None,
            pw:-pw if pw != 0 else None,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
