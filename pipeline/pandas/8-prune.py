#!/usr/bin/env python3
"""
Remove rows with missing Close values from a pandas DataFrame.
"""


def prune(df):
    """
    Remove entries where the Close column has NaN values.

    Args:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: DataFrame without NaN values in Close
    """
    return df.dropna(subset=["Close"])
