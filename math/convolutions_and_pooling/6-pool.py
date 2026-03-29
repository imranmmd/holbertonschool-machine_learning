#!/usr/bin/env python3
"""
Performs pooling on images with max or average pooling.
Supports a kernel of any size and stride.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            m = number of images
            h = image height
            w = image width
            c = number of channels
        kernel_shape (tuple): (kh, kw)
            kh = kernel height
            kw = kernel width
        stride (tuple): (sh, sw)
            sh = stride height
            sw = stride width
        mode (str): 'max' or 'avg'

    Returns:
        numpy.ndarray: pooled images
    """

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w, c))

    # Pooling using two loops: height and width
    for i in range(output_h):
        for j in range(output_w):
            slice_img = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(slice_img, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_img, axis=(1, 2))
            else:
                raise ValueError("Mode must be 'max' or 'avg'")

    return output
