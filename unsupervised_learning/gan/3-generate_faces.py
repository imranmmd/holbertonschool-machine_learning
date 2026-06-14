#!/usr/bin/env python3
"""
Builds a convolutional generator and discriminator for
16x16 grayscale face generation.
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds the convolutional generator and discriminator.

    Returns:
        tuple: (generator, discriminator)
    """

    def get_generator():
        """
        Build the generator model.

        Returns:
            keras.Model: Generator network.
        """

        inputs = keras.Input(shape=(16,))

        x = keras.layers.Dense(
            2 * 2 * 512,
            activation="tanh"
        )(inputs)

        x = keras.layers.Reshape(
            (2, 2, 512)
        )(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            16,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            1,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(
            inputs,
            outputs,
            name="generator"
        )

    def get_discriminator():
        """
        Build the discriminator model.

        Returns:
            keras.Model: Discriminator network.
        """

        inputs = keras.Input(
            shape=(16, 16, 1)
        )

        x = keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        )(inputs)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            256,
            (3, 3),
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(
            1,
            activation="tanh"
        )(x)

        return keras.Model(
            inputs,
            outputs,
            name="discriminator"
        )

    return get_generator(), get_discriminator()
