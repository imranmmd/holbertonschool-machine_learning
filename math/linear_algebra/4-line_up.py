#!/usr/bin/env python3
"""
Module that contains a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    If the arrays are not the same length, the function returns None.

    Args:
        arr1 (list): The first list of numbers.
        arr2 (list): The second list of numbers.

    Returns:
        list or None: A new list containing the element-wise sum of
        arr1 and arr2, or None if the lists are of different lengths.
    """
    if len(arr1) != len(arr2):
        return None

    arr = []
    for i in range(len(arr1)):
        arr.append(arr1[i] + arr2[i])

    return arr
