#!/usr/bin/env python3
import pandas as pd


def from_numpy(arr):
    num_cols = arr.shape[1]
    if num_cols > 26:
        raise ValueError("Array has more than 26 columns")

    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(arr, columns=col_names)
