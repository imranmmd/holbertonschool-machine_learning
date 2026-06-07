#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims: tuple containing input dimensions
        filters: list containing the number of filters for each encoder layer
        latent_dims: tuple containing latent space dimensions

    Returns:
        encoder, decoder, auto
    """

    # ================= Encoder =================
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same"
        )(x)

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=x
    )

    # ================= Decoder =================
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs

    rev_filters = filters[::-1]

    # All decoder layers except the last two
    for f in rev_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)

        x = keras.layers.UpSampling2D(
            size=(2, 2)
        )(x)

    # Second-to-last convolution
    x = keras.layers.Conv2D(
        filters=rev_filters[-1],
        kernel_size=(3, 3),
        padding="valid",
        activation="relu"
    )(x)

    x = keras.layers.UpSampling2D(
        size=(2, 2)
    )(x)

    # Final convolution
    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=outputs
    )

    # ================= Autoencoder =================
    auto_inputs = encoder_inputs
    auto_outputs = decoder(encoder(auto_inputs))

    auto = keras.Model(
        inputs=auto_inputs,
        outputs=auto_outputs
    )

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
