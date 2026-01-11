#!/usr/bin/env python3
"""
Clean and fill missing values in a pandas DataFrame.
"""


def fill(df):
    """
    Clean and fill missing values in the DataFrame.

    The function removes the Weighted_Price column, fills
    missing Close values using the previous row, fills
    missing Open, High, and Low values using the Close value
    from the same row, and sets missing volume values to 0.

    Args:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: Cleaned and filled DataFrame
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
