#!/usr/bin/env python3
"""
Slice specific columns from a pandas DataFrame.
"""

import pandas as pd


def slice(df):
    """
    Extract selected columns and slice every 60th row.

    The function selects the High, Low, Close, and Volume_(BTC)
    columns and returns every 60th row.

    Args:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: Sliced DataFrame
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
