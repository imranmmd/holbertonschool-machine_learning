#!/usr/bin/env python3
"""
Module that contains a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.

    If the matrices cannot be multiplied, the function returns None.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists or None: The product of mat1 and mat2, or
        None if the matrices are incompatible.
    """
    # Check multiplication validity
    if len(mat1[0]) != len(mat2):
        return None

    rows = len(mat1)
    cols = len(mat2[0])
    common = len(mat2)

    result = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
