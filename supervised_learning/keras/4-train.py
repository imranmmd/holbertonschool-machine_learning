#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: model to train
        data: input data of shape (m, nx)
        labels: one-hot labels of shape (m, classes)
        batch_size: size of each batch
        epochs: number of epochs
        verbose: determines if output is printed during training
        shuffle: determines whether to shuffle data every epoch

    Returns:
        The History object generated after training
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
