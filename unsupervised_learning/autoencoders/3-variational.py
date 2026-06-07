#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K


def sampling(args):
    """Reparameterization trick."""
    mu, log_var = args

    epsilon = K.random_normal(
        shape=(K.shape(mu)[0], K.int_shape(mu)[1])
    )

    return mu + K.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: input dimensions
        hidden_layers: encoder hidden layer sizes
        latent_dims: latent space dimensions

    Returns:
        encoder, decoder, auto
    """

    # ================= Encoder =================

    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = keras.layers.Dense(
            units,
            activation="relu"
        )(x)

    mu = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    z = keras.layers.Lambda(
        sampling
    )([mu, log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, mu, log_var]
    )

    # ================= Decoder =================

    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(
            units,
            activation="relu"
        )(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # ================= Autoencoder =================

    outputs = decoder(z)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        outputs
    )

    reconstruction_loss *= input_dims

    # KL divergence loss
    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=-1
    )

    vae_loss = K.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
