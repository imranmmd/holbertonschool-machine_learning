#!/usr/bin/env python3
"""
Batch Normalization Layer for TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Args:
        prev (tensor): activated output of the previous layer
        n (int): number of nodes in the layer to be created
        activation (callable): activation function for the layer

    Returns:
        tensor: activated output of the layer
    """
    # Dense layer with specified initializer
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )(prev)

    # Batch normalization parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    epsilon = 1e-7

    # Batch normalization
    mean, variance = tf.nn.moments(dense_layer, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        dense_layer, mean, variance, beta, gamma, epsilon
    )

    # Apply activation
    return activation(Z_norm)
