#!/usr/bin/env python3
"""
Module that contains a function to perform element-wise operations
(addition, subtraction, multiplication, division) on arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division between two arrays or values.

    Args:
        mat1: First array-like object (numpy.ndarray or compatible)
        mat2: Second array-like object (numpy.ndarray, compatible, or scalar)

    Returns:
        tuple: A tuple containing four numpy arrays corresponding to
        the element-wise sum, difference, product, and quotient.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
