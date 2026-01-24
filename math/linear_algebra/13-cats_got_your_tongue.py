#!/usr/bin/env python3
"""
Module that contains a function to concatenate two numpy arrays
along a specified axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along a specific axis.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int, optional): The axis along which to concatenate.
            Defaults to 0.

    Returns:
        numpy.ndarray: A new array resulting from the concatenation.
    """
    return np.concatenate((mat1, mat2), axis=axis)
