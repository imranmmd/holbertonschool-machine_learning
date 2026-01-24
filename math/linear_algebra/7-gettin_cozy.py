#!/usr/bin/env python3
"""
Module that contains a function to concatenate two 2D matrices
along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.
        axis (int): The axis along which to concatenate:
            0 for rows, 1 for columns.

    Returns:
        list of lists or None: A new matrix resulting from the
        concatenation of mat1 and mat2, or None if the matrices
        cannot be concatenated.
    """
    if axis == 0:
        # Number of columns must match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Number of rows must match
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    return None
