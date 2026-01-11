#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


# Load data
df = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ','
)

# Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set Date as index
df = df.set_index('Date')

# Fill missing Close values with previous value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill Open, High, Low with Close value (same row)
for col in ['Open', 'High', 'Low']:
    df[col] = df[col].fillna(df['Close'])

# Fill volume columns with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Keep data from 2017 onwards
df = df[df.index >= '2017-01-01']

# Resample daily with required aggregations
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print transformed DataFrame
print(df_daily)

# Plot
df_daily.plot(figsize=(12, 8))
plt.show()
