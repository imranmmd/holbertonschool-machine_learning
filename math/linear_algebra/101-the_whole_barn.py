#!/usr/bin/env python3
"""
Module that contains a function to add two matrices of arbitrary depth.
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise, supporting nested lists.

    Args:
        mat1 (list): First matrix (can be nested).
        mat2 (list): Second matrix (same shape as mat1).

    Returns:
        list: New matrix with element-wise sums, or None if shapes differ.
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        res = []
        for a, b in zip(mat1, mat2):
            summed = add_matrices(a, b)
            if summed is None:
                return None
            res.append(summed)
        return res
    else:
        # Base case: mat1 and mat2 are numbers
        return mat1 + mat2
