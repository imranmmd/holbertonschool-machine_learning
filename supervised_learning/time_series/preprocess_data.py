#!/usr/bin/env python3
"""
Preprocess Coinbase and Bitstamp minute-level Bitcoin datasets.

Processing pipeline:
1. Load Coinbase and Bitstamp CSV files.
2. Validate and clean the required columns.
3. Combine exchange observations by timestamp.
4. Fill missing one-minute periods.
5. Aggregate minute observations into hourly candles.
6. Create time-based cyclical features.
7. Split the data chronologically.
8. Standardize features using training statistics only.
9. Save everything in a compressed NPZ file.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Weighted_Price",
]


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Coinbase and Bitstamp BTC data for "
            "next-hour price forecasting."
        )
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        help=(
            "Paths to the raw Coinbase and Bitstamp CSV files. "
            "You may provide one or more files."
        ),
    )

    parser.add_argument(
        "--output",
        default="btc_hourly.npz",
        help="Path of the generated NPZ file.",
    )

    parser.add_argument(
        "--metadata",
        default="btc_hourly_metadata.json",
        help="Path of the generated metadata JSON file.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of observations used for training.",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Fraction of observations used for validation.",
    )

    return parser.parse_args()


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
) -> None:
    """Validate train, validation, and test ratios."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            "train_ratio must be greater than 0 and smaller than 1."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be greater than 0 and smaller than 1."
        )

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be smaller than 1."
        )


def load_exchange_csv(csv_path: str) -> pd.DataFrame:
    """
    Load one Coinbase or Bitstamp CSV file.

    Invalid timestamps are removed. Other columns are converted to numeric
    values, with malformed values becoming NaN.
    """

    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [
        column
        for column in RAW_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing the following columns: "
            f"{missing_columns}"
        )

    dataframe = dataframe[RAW_COLUMNS].copy()

    for column in RAW_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(subset=["Timestamp"])

    dataframe["Timestamp"] = dataframe["Timestamp"].astype(
        np.int64
    )

    return dataframe


def combine_exchanges(
    csv_files: Iterable[str],
) -> pd.DataFrame:
    """
    Combine Coinbase and Bitstamp observations.

    When both exchanges contain the same timestamp:

    - Open is averaged.
    - Close is averaged.
    - High is the maximum.
    - Low is the minimum.
    - Volumes are summed.
    - Weighted price is weighted by BTC transaction volume.
    """

    dataframes = [
        load_exchange_csv(csv_file)
        for csv_file in csv_files
    ]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined["Weighted_Value"] = (
        combined["Weighted_Price"]
        * combined["Volume_(BTC)"]
    )

    grouped = combined.groupby(
        "Timestamp",
        sort=True,
    ).agg(
        Open=("Open", "mean"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "mean"),
        Volume_BTC=("Volume_(BTC)", "sum"),
        Volume_Currency=("Volume_(Currency)", "sum"),
        Weighted_Value=("Weighted_Value", "sum"),
        Weighted_Price_Fallback=("Weighted_Price", "mean"),
    )

    btc_volume = grouped["Volume_BTC"].replace(
        0.0,
        np.nan,
    )

    grouped["Weighted_Price"] = (
        grouped["Weighted_Value"] / btc_volume
    ).fillna(
        grouped["Weighted_Price_Fallback"]
    )

    grouped = grouped.drop(
        columns=[
            "Weighted_Value",
            "Weighted_Price_Fallback",
        ]
    )

    grouped.index = pd.to_datetime(
        grouped.index,
        unit="s",
        utc=True,
    )

    grouped.index.name = "Datetime"

    return grouped.sort_index()


def fill_missing_minutes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a regular one-minute time series.

    Missing prices are interpolated using time interpolation.
    Missing volume is treated as zero transaction volume.
    """

    complete_index = pd.date_range(
        start=dataframe.index.min().floor("min"),
        end=dataframe.index.max().floor("min"),
        freq="min",
        tz="UTC",
    )

    dataframe = dataframe.reindex(
        complete_index
    )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Weighted_Price",
    ]

    dataframe[price_columns] = (
        dataframe[price_columns]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    volume_columns = [
        "Volume_BTC",
        "Volume_Currency",
    ]

    dataframe[volume_columns] = (
        dataframe[volume_columns]
        .fillna(0.0)
    )

    return dataframe


def aggregate_to_hourly(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert minute-level observations into hourly observations.

    This makes the model input contain 24 time steps for the previous
    24 hours instead of 1,440 one-minute time steps.
    """

    hourly = dataframe.resample("h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume_BTC": "sum",
            "Volume_Currency": "sum",
            "Weighted_Price": "mean",
        }
    )

    hourly = hourly.dropna(
        subset=PRICE_COLUMNS
    )

    # Volume distributions are highly skewed.
    # log1p reduces the effect of extreme values.
    hourly["Log_Volume_BTC"] = np.log1p(
        hourly["Volume_BTC"].clip(lower=0.0)
    )

    hourly["Log_Volume_Currency"] = np.log1p(
        hourly["Volume_Currency"].clip(lower=0.0)
    )

    hourly = hourly.drop(
        columns=[
            "Volume_BTC",
            "Volume_Currency",
        ]
    )

    # Raw Unix time is not directly useful.
    # Encode repeated calendar patterns using sine and cosine.
    hour_angle = (
        2.0
        * np.pi
        * hourly.index.hour
        / 24.0
    )

    weekday_angle = (
        2.0
        * np.pi
        * hourly.index.dayofweek
        / 7.0
    )

    hourly["Hour_Sin"] = np.sin(
        hour_angle
    )

    hourly["Hour_Cos"] = np.cos(
        hour_angle
    )

    hourly["Weekday_Sin"] = np.sin(
        weekday_angle
    )

    hourly["Weekday_Cos"] = np.cos(
        weekday_angle
    )

    return hourly.astype(
        np.float32
    )


def standardize_features(
    values: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using statistics from the training period only.

    This avoids data leakage from validation or test observations.
    """

    feature_mean = values[:train_end].mean(
        axis=0
    )

    feature_std = values[:train_end].std(
        axis=0
    )

    feature_std = np.where(
        feature_std < 1e-8,
        1.0,
        feature_std,
    )

    standardized = (
        values - feature_mean
    ) / feature_std

    return (
        standardized.astype(np.float32),
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
    )


def preprocess_data(
    csv_files: Iterable[str],
    output_path: str,
    metadata_path: str,
    train_ratio: float,
    validation_ratio: float,
) -> None:
    """Run the complete preprocessing workflow."""

    validate_split_ratios(
        train_ratio,
        validation_ratio,
    )

    print("Loading and combining exchange data...")

    minute_data = combine_exchanges(
        csv_files
    )

    print(
        f"Combined timestamp rows: "
        f"{len(minute_data):,}"
    )

    print("Filling missing one-minute observations...")

    minute_data = fill_missing_minutes(
        minute_data
    )

    print("Aggregating data into hourly candles...")

    hourly_data = aggregate_to_hourly(
        minute_data
    )

    if len(hourly_data) < 100:
        raise ValueError(
            "Too few hourly observations were produced."
        )

    feature_names = list(
        hourly_data.columns
    )

    feature_values = hourly_data.to_numpy(
        dtype=np.float32
    )

    target_values = hourly_data[
        "Close"
    ].to_numpy(
        dtype=np.float32
    )

    number_of_rows = len(
        hourly_data
    )

    train_end = int(
        number_of_rows * train_ratio
    )

    validation_end = int(
        number_of_rows
        * (
            train_ratio
            + validation_ratio
        )
    )

    scaled_features, feature_mean, feature_std = (
        standardize_features(
            feature_values,
            train_end,
        )
    )

    target_mean = np.float32(
        target_values[:train_end].mean()
    )

    target_std = np.float32(
        target_values[:train_end].std()
    )

    if target_std < 1e-8:
        target_std = np.float32(1.0)

    scaled_target = (
        (
            target_values
            - target_mean
        )
        / target_std
    ).astype(
        np.float32
    )

    timestamps = (
        hourly_data.index.astype("int64")
        // 10**9
    ).to_numpy(
        dtype=np.int64
    )

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        features=scaled_features,
        target=scaled_target,
        timestamps=timestamps,
        train_end=np.int64(train_end),
        validation_end=np.int64(validation_end),
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        feature_names=np.asarray(
            feature_names
        ),
    )

    metadata = {
        "source_files": [
            str(Path(csv_file))
            for csv_file in csv_files
        ],
        "output_file": str(output_file),
        "number_of_rows": number_of_rows,
        "start_time_utc": (
            hourly_data.index[0].isoformat()
        ),
        "end_time_utc": (
            hourly_data.index[-1].isoformat()
        ),
        "train_rows": train_end,
        "validation_rows": (
            validation_end - train_end
        ),
        "test_rows": (
            number_of_rows - validation_end
        ),
        "feature_names": feature_names,
        "target": (
            "Bitcoin close price at the end "
            "of the following hour"
        ),
        "sampling_frequency": "1 hour",
        "lookback_hours": 24,
        "scaling": (
            "Z-score standardization using "
            "training-period statistics only"
        ),
    }

    metadata_file = Path(
        metadata_path
    )

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nPreprocessing completed.")
    print(
        f"Total hourly rows: "
        f"{number_of_rows:,}"
    )
    print(
        f"Training rows: "
        f"{train_end:,}"
    )
    print(
        f"Validation rows: "
        f"{validation_end - train_end:,}"
    )
    print(
        f"Test rows: "
        f"{number_of_rows - validation_end:,}"
    )
    print(
        f"Saved dataset to: "
        f"{output_file}"
    )
    print(
        f"Saved metadata to: "
        f"{metadata_file}"
    )


def main() -> None:
    """Run preprocessing from the command line."""

    arguments = parse_arguments()

    preprocess_data(
        csv_files=arguments.csv_files,
        output_path=arguments.output,
        metadata_path=arguments.metadata,
        train_ratio=arguments.train_ratio,
        validation_ratio=arguments.validation_ratio,
    )


if __name__ == "__main__":
    main()
