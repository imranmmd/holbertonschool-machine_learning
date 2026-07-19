#!/usr/bin/env python3
"""Creates and trains a Gensim Word2Vec model."""

import gensim


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
    Create, build, and train a Gensim Word2Vec model.

    Args:
        sentences: List of tokenized sentences to train on.
        vector_size: Dimensionality of the word embeddings.
        min_count: Minimum number of occurrences required for a word.
        window: Maximum distance between current and predicted words.
        negative: Number of negative samples.
        cbow: Use CBOW if True and Skip-gram if False.
        epochs: Number of training iterations.
        seed: Seed for the random number generator.
        workers: Number of worker threads used for training.

    Returns:
        The trained Gensim Word2Vec model.
    """
    model = gensim.models.Word2Vec(
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

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    return model
