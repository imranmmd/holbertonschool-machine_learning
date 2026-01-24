#!/usr/bin/env python3
"""
Module that contains a function to add two 2D matrices.
"""


def add_matrices2D(arr1, arr2):
    """
    Adds two 2D matrices element-wise.

    If the matrices are not the same shape, the function returns None.

    Args:
        arr1 (list of lists): First 2D matrix.
        arr2 (list of lists): Second 2D matrix.

    Returns:
        list of lists or None: A new matrix containing the element-wise
        sum of arr1 and arr2, or None if the matrices are not the same shape.
    """
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        return None

    result = [[0] * len(arr1[0]) for _ in range(len(arr1))]

    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            result[i][j] = arr1[i][j] + arr2[i][j]

    return result
