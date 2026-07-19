#!/usr/bin/env python3
"""Create and train a Gensim FastText model."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a Gensim FastText model.

    Args:
        sentences: List of tokenized sentences used for training.
        vector_size: Dimensionality of the word embeddings.
        min_count: Minimum number of occurrences required for a word.
        negative: Number of negative samples used during training.
        window: Maximum distance between target and context words.
        cbow: If True, use CBOW; otherwise, use Skip-gram.
        epochs: Number of training iterations over the corpus.
        seed: Seed for the random number generator.
        workers: Number of worker threads used during training.

    Returns:
        The trained Gensim FastText model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg,
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
