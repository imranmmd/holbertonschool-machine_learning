#!/usr/bin/env python3
"""
Module that contains a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    The transpose of a matrix is obtained by flipping the matrix
    over its diagonal, switching rows and columns.

    Args:
        matrix (list of lists): A 2D matrix to transpose.

    Returns:
        list of lists: A new matrix representing the transpose.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
