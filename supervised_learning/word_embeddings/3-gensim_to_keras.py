#!/usr/bin/env python3
"""Convert a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Convert a trained Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer initialized with the
        Word2Vec model's vectors.
    """
    embedding_matrix = model.wv.vectors

    vocab_size, embedding_dim = embedding_matrix.shape

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return layer
