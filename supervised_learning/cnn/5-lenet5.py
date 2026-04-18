#!/usr/bin/env python3
"""LeNet-5 architecture with Keras"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 model

    Parameters:
    X: K.Input of shape (m, 28, 28, 1)

    Returns:
    Compiled Keras model
    """
    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv
    A = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Layer 2: Max Pool
    A = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(A)

    # Layer 3: Conv
    A = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(A)

    # Layer 4: Max Pool
    A = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(A)

    # Flatten
    A = K.layers.Flatten()(A)

    # Fully connected layers
    A = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(A)

    A = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(A)

    # Output layer
    outputs = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(A)

    # Build model
    model = K.Model(inputs=X, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
