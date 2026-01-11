#!/usr/bin/env python3
"""
Convert selected DataFrame columns to a NumPy array.
"""

import pandas as pd


def array(df):
    """
    Select the last 10 rows of High and Close columns and
    convert them to a NumPy array.

    Args:
        df (pandas.DataFrame): Input DataFrame containing
        High and Close columns

    Returns:
        numpy.ndarray: Array of selected values
    """
    return df[["High", "Close"]].tail(10).to_numpy()
