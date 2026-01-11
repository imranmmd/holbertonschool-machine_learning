#!/usr/bin/env python3
import pandas as pd
"""Convert a 2D NumPy array to a Pandas DataFrame with column names as uppercase letters."""


def from_numpy(arr):
    """Convert a 2D NumPy array to a Pandas DataFrame with column names as uppercase letters."""
    num_cols = arr.shape[1]
    if num_cols > 26:
        raise ValueError("Array has more than 26 columns")

    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(arr, columns=col_names)
