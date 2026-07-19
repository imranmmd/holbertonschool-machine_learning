#!/usr/bin/env python3
"""Creates a bag-of-words embedding matrix."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: List of vocabulary words to use. If None, all words
            found in sentences are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f), where:
            s is the number of sentences
            f is the number of features
        features: numpy.ndarray containing the vocabulary features
            used in the embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    features = vectorizer.get_feature_names_out()

    return embeddings, features
