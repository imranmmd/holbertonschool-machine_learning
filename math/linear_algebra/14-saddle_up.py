#!/usr/bin/env python3
"""
Module that contains a function to perform matrix multiplication
on numpy arrays.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication between two numpy arrays.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The product of mat1 and mat2.
    """
    return np.matmul(mat1, mat2)
