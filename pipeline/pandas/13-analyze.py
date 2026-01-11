#!/usr/bin/env python3
"""
Compute descriptive statistics for a DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: Descriptive statistics DataFrame
    """
    # Drop Timestamp column
    df = df.drop(columns=['Timestamp'])

    # Compute descriptive statistics
    return df.describe()
