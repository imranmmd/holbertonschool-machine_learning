#!/usr/bin/env python3
"""
Concatenate bitstamp and coinbase DataFrames with labeled keys.
"""

index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both DataFrames on Timestamp, select bitstamp rows up to a
    specific timestamp, and concatenate them above coinbase data.

    Args:
        df1 (pandas.DataFrame): Coinbase DataFrame
        df2 (pandas.DataFrame): Bitstamp DataFrame

    Returns:
        pandas.DataFrame: Concatenated DataFrame with keys
    """
    # Index both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Select bitstamp data up to and including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
