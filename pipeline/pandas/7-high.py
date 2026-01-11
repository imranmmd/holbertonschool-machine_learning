#!/usr/bin/env python3
"""
Sort a pandas DataFrame by the High price in descending order.
"""


def high(df):
    """
    Sort the DataFrame by the High column in descending order.

    Args:
        df (pandas.DataFrame): Input DataFrame with High column

    Returns:
        pandas.DataFrame: DataFrame sorted by High descending
    """
    return df.sort_values(by="High", ascending=False)
