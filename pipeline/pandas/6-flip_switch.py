#!/usr/bin/env python3
"""
Sort and transpose a pandas DataFrame.
"""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order
    and transpose it.

    Args:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: Transformed DataFrame
    """
    df = df.sort_index(ascending=False)
    return df.T
