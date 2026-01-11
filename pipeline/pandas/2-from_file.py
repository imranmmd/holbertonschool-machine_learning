#!/usr/bin/env python3
"""
Load data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file into a pandas DataFrame.

    Args:
        filename (str): Path to the file to load
        delimiter (str): Column separator used in the file

    Returns:
        pandas.DataFrame: Loaded DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
