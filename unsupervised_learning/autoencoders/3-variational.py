#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def sampling(args, latent_dims):
    """Samples a latent vector using the reparameterization trick."""
    mu, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dims))
    return mu + K.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    latent = keras.layers.Lambda(
        lambda t: sampling(t, latent_dims)
    )([mu, log_var])

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=[latent, mu, log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs
    )

    # Autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    latent, mu, log_var = encoder(auto_inputs)
    auto_outputs = decoder(latent)

    auto = keras.Model(
        inputs=auto_inputs,
        outputs=auto_outputs
    )

    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=1
    )
    auto.add_loss(K.mean(kl_loss))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
