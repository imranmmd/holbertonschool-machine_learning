#!/usr/bin/env python3
"""
Module for performing convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): shape (m, h, w)
        kernel (numpy.ndarray): shape (kh, kw)
        padding (tuple): (ph, pw)

    Returns:
        numpy.ndarray: convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    output = np.zeros((m, output_h, output_w))

    # Convolution (ONLY 2 loops)
    for i in range(output_h):
        for j in range(output_w):
            slice_img = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2))

    return output
