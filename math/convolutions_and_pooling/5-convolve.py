#!/usr/bin/env python3
"""
Performs a convolution on multi-channel images using multiple kernels.
Supports 'same', 'valid', or custom padding with a stride.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multi-channel images with multiple kernels.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            m = number of images
            h = image height
            w = image width
            c = number of channels
        kernels (numpy.ndarray): shape (kh, kw, c, nc)
            kh = kernel height
            kw = kernel width
            nc = number of kernels
        padding (tuple or str): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: convolved images, shape (m, output_h, output_w, nc)
    """

    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Determine padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Compute output dimensions
    output_h = ((padded.shape[1] - kh) // sh) + 1
    output_w = ((padded.shape[2] - kw) // sw) + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w, nc))

    # Convolution using 3 loops: height, width, kernels
    for i in range(output_h):
        for j in range(output_w):
            slice_img = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    slice_img * kernels[:, :, :, k], axis=(1, 2, 3)
                )

    return output
