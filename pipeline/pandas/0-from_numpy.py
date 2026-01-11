#!/usr/bin/env python3

import pandas as pd
import numpy as np

def from_numpy(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    num_cols = arr.shape[1]
    if num_cols > 26:
        raise ValueError("Array has more than 26 columns")

    col_names = [chr(65 + i) for i in range(num_cols)]
    df = pd.DataFrame(arr, columns=col_names)

    return df
