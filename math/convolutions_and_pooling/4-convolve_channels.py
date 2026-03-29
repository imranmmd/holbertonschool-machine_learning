#!/usr/bin/env python3
"""
Performs a convolution on multi-channel images using a single kernel.
Supports 'same', 'valid', or custom padding with a stride.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multi-channel images.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            m = number of images
            h = image height
            w = image width
            c = number of channels
        kernel (numpy.ndarray): shape (kh, kw, c)
            kh = kernel height
            kw = kernel width
        padding (tuple or str): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: convolved images, shape (m, output_h, output_w)
    """

    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
    output = np.zeros((m, output_h, output_w))

    # Convolution using only 2 loops
    for i in range(output_h):
        for j in range(output_w):
            vert_start = i * sh
            horiz_start = j * sw
            slice_img = padded[
                :, vert_start:vert_start+kh, horiz_start:horiz_start+kw, :
            ]
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2, 3))

    return output
