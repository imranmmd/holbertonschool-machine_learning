#!/usr/bin/env python3
"""
Module that contains a function to calculate the shape of a matrix.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    The shape is returned as a list of integers where each integer
    represents the size of the matrix in that dimension.

    Args:
        matrix (list): A nested list representing a matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
