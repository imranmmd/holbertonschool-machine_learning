#!/usr/bin/env python3
"""
Create a hierarchical DataFrame indexed by Timestamp first.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate bitstamp and coinbase data between specific timestamps,
    rearranging the MultiIndex so Timestamp is the first level.

    Args:
        df1 (pandas.DataFrame): Coinbase DataFrame
        df2 (pandas.DataFrame): Bitstamp DataFrame

    Returns:
        pandas.DataFrame: Hierarchically indexed DataFrame
    """
    # Index both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Slice required timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Swap MultiIndex levels so Timestamp is first
    df = df.swaplevel(0, 1)

    # Sort chronologically by Timestamp
    df = df.sort_index()

    return df
