#!/usr/bin/env python3
"""
Convert a 2D NumPy array to a Pandas DataFrame.

The DataFrame uses uppercase letters as column names.
"""

import pandas as pd


def from_numpy(arr):
    """
    Convert a 2D NumPy array to a Pandas DataFrame.

    The DataFrame uses uppercase letters as column names.
    """
    num_cols = arr.shape[1]
    if num_cols > 26:
        raise ValueError("Array has more than 26 columns")

    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(arr, columns=col_names)
