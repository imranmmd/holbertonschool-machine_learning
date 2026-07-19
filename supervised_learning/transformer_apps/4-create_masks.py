#!/usr/bin/env python3
"""Create masks used by a Transformer network."""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Create encoder, combined, and decoder masks.

    Args:
        inputs: Tensor of shape (batch_size, seq_len_in).
        target: Tensor of shape (batch_size, seq_len_out).

    Returns:
        encoder_mask: Padding mask for the encoder.
        combined_mask: Padding and look-ahead mask for the decoder.
        decoder_mask: Padding mask for encoder-decoder attention.
    """
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0),
        tf.float32
    )
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0),
        tf.float32
    )
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    target_padding_mask = tf.cast(
        tf.math.equal(target, 0),
        tf.float32
    )
    target_padding_mask = target_padding_mask[
        :, tf.newaxis, tf.newaxis, :
    ]

    target_length = tf.shape(target)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target_length, target_length)),
        -1,
        0
    )
    look_ahead_mask = look_ahead_mask[
        tf.newaxis, tf.newaxis, :, :
    ]

    combined_mask = tf.maximum(
        target_padding_mask,
        look_ahead_mask
    )

    return encoder_mask, combined_mask, decoder_mask
