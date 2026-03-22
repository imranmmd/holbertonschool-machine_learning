#!/usr/bin/env python3
"""Builds a neural network with Keras using the Functional API"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    Args:
        nx: number of input features
        layers: list containing number of nodes in each layer
        activations: list containing activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
