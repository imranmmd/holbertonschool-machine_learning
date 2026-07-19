#!/usr/bin/env python3
"""Create and train a Gensim Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a Gensim Word2Vec model.

    Args:
        sentences: List of tokenized sentences to train on.
        vector_size: Dimensionality of the word embeddings.
        min_count: Minimum occurrences required to include a word.
        window: Maximum distance between target and context words.
        negative: Number of negative samples used during training.
        cbow: If True, use CBOW; otherwise, use Skip-gram.
        epochs: Number of training iterations over the corpus.
        seed: Seed for the random number generator.
        workers: Number of worker threads used during training.

    Returns:
        The trained Gensim Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=int(not cbow),
        epochs=epochs,
        seed=seed,
        workers=workers,
    )

    model.build_vocab(sentences)

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    return model
