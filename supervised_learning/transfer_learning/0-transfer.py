#!/usr/bin/env python3
"""
0-transfer.py

Train a transfer learning CNN on CIFAR-10 using Keras Applications.

Requirements:
- Uses Keras Applications pretrained model
- Saves compiled model as cifar10.h5
- Includes preprocess_data(X, Y)
- Does not run training when imported
"""

import tensorflow as tf
from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocess CIFAR-10 data

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3)
        Y: numpy.ndarray of shape (m,)

    Returns:
        X_p, Y_p
    """
    X_p = X.astype("float32") / 255.0

    if len(Y.shape) == 2:
        Y = Y.reshape(-1)

    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


def build_model():
    """
    Build transfer learning model using MobileNetV2
    """

    inputs = K.Input(shape=(32, 32, 3))

    # Resize CIFAR images to 96x96
    x = K.layers.Lambda(
        lambda image: tf.image.resize(image, (96, 96))
    )(inputs)

    # MobileNetV2 preprocessing
    x = K.applications.mobilenet_v2.preprocess_input(x * 255.0)

    # Pretrained backbone
    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    model = K.Model(inputs, outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def train():
    """
    Train and save model
    """

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    model, base_model = build_model()

    callbacks = [
        K.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            "cifar10.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # Stage 1: Train classifier head
    model.fit(
        X_train,
        Y_train,
        batch_size=128,
        epochs=15,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Stage 2: Fine tune top layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        Y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        verbose=1
    )

    model.save("cifar10.h5")


if __name__ == "__main__":
    train()
