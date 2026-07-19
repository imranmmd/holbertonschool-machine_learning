#!/usr/bin/env python3
"""Calculate the cumulative n-gram BLEU score."""

import math


def cumulative_bleu(references, sentence, n):
    """
    Calculate the cumulative n-gram BLEU score for a sentence.

    Args:
        references: List of reference translations.
        sentence: Proposed translation represented as a list of words.
        n: Size of the largest n-gram used for evaluation.

    Returns:
        The cumulative n-gram BLEU score.
    """
    sentence_length = len(sentence)

    if sentence_length == 0 or n <= 0 or sentence_length < n:
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

    log_precision_sum = 0

    for size in range(1, n + 1):
        sentence_ngrams = [
            tuple(sentence[index:index + size])
            for index in range(sentence_length - size + 1)
        ]

        clipped_count = 0

        for ngram in set(sentence_ngrams):
            sentence_count = sentence_ngrams.count(ngram)
            maximum_reference_count = 0

            for reference in references:
                reference_ngrams = [
                    tuple(reference[index:index + size])
                    for index in range(len(reference) - size + 1)
                ]

                reference_count = reference_ngrams.count(ngram)

                maximum_reference_count = max(
                    maximum_reference_count,
                    reference_count,
                )

            clipped_count += min(
                sentence_count,
                maximum_reference_count,
            )

        precision = clipped_count / len(sentence_ngrams)

        if precision == 0:
            return 0

        log_precision_sum += math.log(precision)

    geometric_mean = math.exp(
        log_precision_sum / n
    )

    if sentence_length > closest_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(
            1 - closest_length / sentence_length
        )

    return brevity_penalty * geometric_mean
