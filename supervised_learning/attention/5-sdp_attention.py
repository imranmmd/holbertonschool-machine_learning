#!/usr/bin/env python3
"""Calculate scaled dot-product attention."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate scaled dot-product attention.

    Args:
        Q: Query tensor with shape (..., seq_len_q, dk).
        K: Key tensor with shape (..., seq_len_v, dk).
        V: Value tensor with shape (..., seq_len_v, dv).
        mask: Optional tensor broadcastable to
            (..., seq_len_q, seq_len_v).

    Returns:
        output: Tensor with shape (..., seq_len_q, dv).
        weights: Tensor with shape (..., seq_len_q, seq_len_v).
    """
    scores = tf.matmul(
        Q,
        K,
        transpose_b=True
    )

    dk = tf.cast(
        tf.shape(K)[-1],
        tf.float32
    )

    scaled_scores = scores / tf.math.sqrt(dk)

    if mask is not None:
        scaled_scores += mask * -1e9

    weights = tf.nn.softmax(
        scaled_scores,
        axis=-1
    )

    output = tf.matmul(
        weights,
        V
    )

    return output, weights
