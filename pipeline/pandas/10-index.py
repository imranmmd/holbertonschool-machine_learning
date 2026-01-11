#!/usr/bin/env python3
"""
Set the Timestamp column as the index of a pandas DataFrame.
"""


def index(df):
    """
    Set the Timestamp column as the DataFrame index.

    Args:
        df (pandas.DataFrame): Input DataFrame containing Timestamp

    Returns:
        pandas.DataFrame: DataFrame indexed by Timestamp
    """
    return df.set_index("Timestamp")
