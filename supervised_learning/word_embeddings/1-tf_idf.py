#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: List of vocabulary words to use. If None, all words
            found in sentences are used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the
            TF-IDF embeddings.
        features: numpy.ndarray containing the features used.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()

    features = vectorizer.get_feature_names_out()

    return embeddings, features
