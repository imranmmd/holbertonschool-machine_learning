#!/usr/bin/env python3
"""Creates and trains a Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """
    Creates, builds, and trains a Gensim Word2Vec model.

    Args:
        sentences: List of tokenized sentences used for training.
        vector_size: Dimensionality of each word embedding.
        min_count: Minimum number of occurrences required for a word.
        window: Maximum distance between a target word and context words.
        negative: Number of negative samples used during training.
        cbow: If True, uses CBOW. Otherwise, uses Skip-gram.
        epochs: Number of training iterations over the corpus.
        seed: Seed used by the random number generator.
        workers: Number of worker threads used for training.

    Returns:
        The trained Gensim Word2Vec model.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=not cbow,
        epochs=epochs,
        seed=seed,
        workers=workers,
    )

    return model
