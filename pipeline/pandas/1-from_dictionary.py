#!/usr/bin/env python3
import pandas as pd
"""Create a sample Pandas DataFrame with specific data and index."""

df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
    },
    index=["A", "B", "C", "D"]
)
