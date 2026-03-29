#!/usr/bin/env python3
"""
Module for performing valid convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w) containing grayscale images
        kernel (numpy.ndarray): shape (kh, kw) convolution kernel

    Returns:
        numpy.ndarray: convolved images of shape (m, h - kh + 1, w - kw + 1)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            slice_img = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2))

    return output
