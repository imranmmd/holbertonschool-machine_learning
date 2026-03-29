#!/usr/bin/env python3
"""
Module for performing same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)

    Returns:
        numpy.ndarray: convolved images of shape (m, h, w)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Padding
    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            slice_img = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2))

    return output
