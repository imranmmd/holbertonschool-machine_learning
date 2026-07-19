#!/usr/bin/env python3
"""Convert a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Convert a trained Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer initialized with the
        Word2Vec model's vectors.
    """
    weights = model.wv.vectors

    embedding = keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )

    return embedding
