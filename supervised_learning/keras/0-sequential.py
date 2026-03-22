#!/usr/bin/env python3
"""Builds a neural network with Keras"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library

    Args:
        nx: number of input features
        layers: list with number of nodes in each layer
        activations: list with activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability of keeping a node during dropout

    Returns:
        keras Sequential model
    """
    model = K.models.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
