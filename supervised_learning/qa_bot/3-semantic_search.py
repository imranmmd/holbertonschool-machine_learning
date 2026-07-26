#!/usr/bin/env python3
"""Semantic search over a corpus of reference documents."""

import os
import numpy as np
import tensorflow_hub as hub


MODEL = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def semantic_search(corpus_path, sentence):
    """Return the corpus document most semantically similar to a sentence.

    Args:
        corpus_path (str): Path to the directory containing the documents.
        sentence (str): Query sentence used for semantic search.

    Returns:
        str: Text of the document most similar to the query sentence.
    """
    documents = []

    for filename in sorted(os.listdir(corpus_path)):
        file_path = os.path.join(corpus_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())

    embeddings = MODEL([sentence] + documents).numpy()

    query_embedding = embeddings[0]
    document_embeddings = embeddings[1:]

    similarities = np.inner(document_embeddings, query_embedding)
    best_match = np.argmax(similarities)

    return documents[best_match]
