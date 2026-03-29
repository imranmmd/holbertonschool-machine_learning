#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    Arguments:
    images -- numpy.ndarray of shape (m, h, w)
    kernel -- numpy.ndarray of shape (kh, kw)

    Returns:
    output -- numpy.ndarray containing convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Slide kernel
    for i in range(output_h):
        for j in range(output_w):
            # Extract slice
            slice_img = images[:, i:i+kh, j:j+kw]

            # Element-wise multiply + sum
            output[:, i, j] = np.sum(slice_img * kernel, axis=(1, 2))

    return output
