#!/usr/bin/env python3
"""
Module for performing convolution on grayscale images with padding and stride.
Supports 'valid', 'same', or custom padding.
"""

import numpy as np
from math import ceil


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w)
            m = number of images
            h = height
            w = width
        kernel (numpy.ndarray): shape (kh, kw)
            kh = kernel height
            kw = kernel width
        padding (tuple or str): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: convolved images, shape (m, output_h, output_w)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = ceil(((h - 1) * sh + kh - h) / 2)
        pw = ceil(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output dimensions
    output_h = ((padded.shape[1] - kh) // sh) + 1
    output_w = ((padded.shape[2] - kw) // sw) + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Convolution (2 loops)
    for i in range(output_h):
        for j in range(output_w):
            vert_start = i * sh
            horiz_start = j * sw
            slice_img = padded[:, vert_start:vert_start+kh, horiz_start:horiz_start+kw]
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2))

    return output
