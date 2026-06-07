#!/usr/bin/env python3
"""Vanilla Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): dimensions of the model input.
        hidden_layers (list): number of nodes for each hidden layer
                              in the encoder.
        latent_dims (int): dimensions of the latent space.

    Returns:
        encoder, decoder, auto
    """

    # ================= Encoder =================
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(
            units=nodes,
            activation='relu'
        )(x)

    latent = keras.layers.Dense(
        units=latent_dims,
        activation='relu'
    )(x)

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=latent
    )

    # ================= Decoder =================
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            units=nodes,
            activation='relu'
        )(x)

    decoder_outputs = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs
    )

    # ================= Autoencoder =================
    auto_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)

    auto = keras.Model(
        inputs=auto_inputs,
        outputs=decoded
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
