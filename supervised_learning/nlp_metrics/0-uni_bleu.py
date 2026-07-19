#!/usr/bin/env python3
"""Calculate the unigram BLEU score."""

import math


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a proposed sentence.

    Args:
        references: List of reference translations, where each
            translation is represented as a list of words.
        sentence: Proposed translation represented as a list of words.

    Returns:
        The unigram BLEU score.
    """
    sentence_length = len(sentence)

    if sentence_length == 0:
        return 0

    reference_lengths = [
        len(reference)
        for reference in references
    ]

    closest_length = min(
        reference_lengths,
        key=lambda length: (
            abs(length - sentence_length),
            length,
        ),
    )

    clipped_count = 0

    for word in set(sentence):
        sentence_count = sentence.count(word)

        maximum_reference_count = max(
            reference.count(word)
            for reference in references
        )

        clipped_count += min(
            sentence_count,
            maximum_reference_count,
        )

    precision = clipped_count / sentence_length

    if precision == 0:
        return 0

    if sentence_length > closest_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(
            1 - closest_length / sentence_length
        )

    return brevity_penalty * precision
