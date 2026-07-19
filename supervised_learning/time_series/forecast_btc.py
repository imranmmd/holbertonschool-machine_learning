#!/usr/bin/env python3
"""
Train and evaluate a Keras RNN for Bitcoin price forecasting.

The model uses the previous 24 hourly Bitcoin observations to predict
the closing price at the end of the immediately following hour.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


LOOKBACK_HOURS = 24


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a recurrent neural network to predict "
            "the next hourly Bitcoin close price."
        )
    )

    parser.add_argument(
        "--data",
        default="btc_hourly.npz",
        help="Path to the preprocessed NPZ dataset.",
    )

    parser.add_argument(
        "--model-output",
        default="btc_forecaster.keras",
        help="Path used to save the best Keras model.",
    )

    parser.add_argument(
        "--history-output",
        default="training_history.json",
        help="Path used to save training history and metrics.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial Adam learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def load_preprocessed_data(
    file_path: str,
) -> Dict[str, np.ndarray]:
    """Load arrays from the preprocessing output."""

    path = Path(
        file_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}\n"
            "Run preprocess_data.py before forecast_btc.py."
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        data = {
            key: archive[key]
            for key in archive.files
        }

    required_keys = {
        "features",
        "target",
        "timestamps",
        "train_end",
        "validation_end",
        "target_mean",
        "target_std",
    }

    missing_keys = (
        required_keys
        - set(data.keys())
    )

    if missing_keys:
        raise ValueError(
            "The NPZ file is missing the following keys: "
            f"{sorted(missing_keys)}"
        )

    return data


def create_window_dataset(
    features: np.ndarray,
    target: np.ndarray,
    target_start: int,
    target_end: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset containing 24-hour sequences.

    For a target located at index t:

        Input:
            features[t - 24 : t]

        Label:
            target[t]

    Therefore, the model receives the previous 24 complete hourly
    observations and predicts the close price for the following hour.
    """

    if target_start < LOOKBACK_HOURS:
        raise ValueError(
            "target_start must be at least LOOKBACK_HOURS."
        )

    if target_end > len(features):
        raise ValueError(
            "target_end exceeds the number of observations."
        )

    if target_start >= target_end:
        raise ValueError(
            "The requested dataset split is empty."
        )

        window_features = features[
        target_start - LOOKBACK_HOURS:
        target_end - 1
    ]

    window_targets = target[
        target_start:
        target_end
    ]

    dataset = (
        tf.keras.utils.timeseries_dataset_from_array(
            data=window_features,
            targets=window_targets,
            sequence_length=LOOKBACK_HOURS,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=shuffle,
            seed=seed,
            batch_size=batch_size,
        )
    )

    dataset = dataset.cache()

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def build_model(
    number_of_features: int,
    learning_rate: float,
) -> tf.keras.Model:
    """
    Create a stacked GRU regression model.

    GRU is an RNN architecture designed to learn relationships across
    time while reducing the vanishing-gradient problem of simple RNNs.
    """

    inputs = tf.keras.Input(
        shape=(
            LOOKBACK_HOURS,
            number_of_features,
        ),
        name="past_24_hours",
    )

    x = tf.keras.layers.GRU(
        units=64,
        return_sequences=True,
        dropout=0.15,
        name="gru_layer_1",
    )(inputs)

    x = tf.keras.layers.GRU(
        units=32,
        return_sequences=False,
        dropout=0.15,
        name="gru_layer_2",
    )(x)

    x = tf.keras.layers.Dense(
        units=32,
        activation="relu",
        name="dense_hidden",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.15,
        name="dense_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation=None,
        name="next_hour_close",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="btc_next_hour_forecaster",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            ),
        ],
    )

    return model


def calculate_usd_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    target_mean: float,
    target_std: float,
) -> Tuple[float, float]:
    """
    Calculate MAE and RMSE after converting predictions back to USD.
    """

    scaled_predictions = model.predict(
        dataset,
        verbose=0,
    ).reshape(-1)

    scaled_targets = np.concatenate(
        [
            batch_targets.numpy().reshape(-1)
            for _, batch_targets in dataset
        ],
        axis=0,
    )

    predictions_usd = (
        scaled_predictions * target_std
        + target_mean
    )

    targets_usd = (
        scaled_targets * target_std
        + target_mean
    )

    errors = (
        predictions_usd
        - targets_usd
    )

    mae_usd = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse_usd = float(
        np.sqrt(
            np.mean(
                np.square(errors)
            )
        )
    )

    return mae_usd, rmse_usd


def main() -> None:
    """Train, validate, evaluate, and save the model."""

    arguments = parse_arguments()

    if arguments.epochs < 1:
        raise ValueError(
            "epochs must be a positive integer."
        )

    if arguments.batch_size < 1:
        raise ValueError(
            "batch-size must be a positive integer."
        )

    if arguments.learning_rate <= 0:
        raise ValueError(
            "learning-rate must be positive."
        )

    tf.keras.utils.set_random_seed(
        arguments.seed
    )

    data = load_preprocessed_data(
        arguments.data
    )

    features = data[
        "features"
    ].astype(
        np.float32
    )

    target = data[
        "target"
    ].astype(
        np.float32
    )

    train_end = int(
        data["train_end"]
    )

    validation_end = int(
        data["validation_end"]
    )

    target_mean = float(
        data["target_mean"]
    )

    target_std = float(
        data["target_std"]
    )

    if train_end <= LOOKBACK_HOURS:
        raise ValueError(
            "The training split is too short for "
            "a 24-hour input window."
        )

    if validation_end >= len(features):
        raise ValueError(
            "The test split is empty."
        )

    print("Creating TensorFlow datasets...")

    train_dataset = create_window_dataset(
        features=features,
        target=target,
        target_start=LOOKBACK_HOURS,
        target_end=train_end,
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=arguments.seed,
    )

    validation_dataset = create_window_dataset(
        features=features,
        target=target,
        target_start=train_end,
        target_end=validation_end,
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=arguments.seed,
    )

    test_dataset = create_window_dataset(
        features=features,
        target=target,
        target_start=validation_end,
        target_end=len(features),
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=arguments.seed,
    )

    number_of_features = features.shape[1]

    model = build_model(
        number_of_features=number_of_features,
        learning_rate=arguments.learning_rate,
    )

    model.summary()

    model_output_path = Path(
        arguments.model_output
    )

    model_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting model training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=arguments.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nLoading best validation checkpoint...")

    best_model = tf.keras.models.load_model(
        model_output_path
    )

    print("\nEvaluating the model on the test set...")

    normalized_metrics = best_model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=2,
    )

    mae_usd, rmse_usd = calculate_usd_metrics(
        model=best_model,
        dataset=test_dataset,
        target_mean=target_mean,
        target_std=target_std,
    )

    print("\nTest results")
    print("-" * 50)

    print(
        "Normalized MSE:  "
        f"{normalized_metrics['loss']:.6f}"
    )

    print(
        "Normalized MAE:  "
        f"{normalized_metrics['mae']:.6f}"
    )

    print(
        "Normalized RMSE: "
        f"{normalized_metrics['rmse']:.6f}"
    )

    print(
        "MAE in USD:      "
        f"${mae_usd:,.2f}"
    )

    print(
        "RMSE in USD:     "
        f"${rmse_usd:,.2f}"
    )

    report = {
        "history": {
            metric_name: [
                float(value)
                for value in metric_values
            ]
            for metric_name, metric_values
            in history.history.items()
        },
        "test_metrics_normalized": {
            metric_name: float(metric_value)
            for metric_name, metric_value
            in normalized_metrics.items()
        },
        "test_mae_usd": mae_usd,
        "test_rmse_usd": rmse_usd,
        "lookback_hours": LOOKBACK_HOURS,
        "number_of_features": number_of_features,
        "target": (
            "Bitcoin close price at the end "
            "of the immediately following hour"
        ),
        "saved_model": str(
            model_output_path
        ),
    }

    history_output_path = Path(
        arguments.history_output
    )

    history_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_output_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved best model to: "
        f"{model_output_path}"
    )

    print(
        "Saved training report to: "
        f"{history_output_path}"
    )


if __name__ == "__main__":
    main()
