#!/usr/bin/env python3
"""
Module that contains a function to slice a numpy array along
specified axes using a dictionary of slices.
"""


def np_slice(matrix, axes={}):
    """
    Slices a numpy array along specific axes.

    Args:
        matrix (numpy.ndarray): The input array to slice.
        axes (dict): A dictionary where the key is the axis to slice along,
                     and the value is a tuple representing the slice
                     (start, stop, step), where start/stop/step are optional.

    Returns:
        numpy.ndarray: The sliced array.
    """
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
