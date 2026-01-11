#!/usr/bin/env python3
"""
Rename and process timestamp data in a pandas DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime and convert it.

    The function converts timestamp values to datetime objects
    and returns only the Datetime and Close columns.

    Args:
        df (pandas.DataFrame): Input DataFrame containing Timestamp

    Returns:
        pandas.DataFrame: Modified DataFrame with Datetime and Close
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df[["Datetime", "Close"]]
