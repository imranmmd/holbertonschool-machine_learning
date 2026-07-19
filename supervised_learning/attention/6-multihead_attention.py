#!/usr/bin/env python3
"""Defines a multi-head attention layer."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head scaled dot-product attention."""

    def __init__(self, dm, h):
        """
        Initialize the multi-head attention layer.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Perform multi-head attention.

        Args:
            Q: Tensor of shape (batch, seq_len_q, dk).
            K: Tensor of shape (batch, seq_len_v, dk).
            V: Tensor of shape (batch, seq_len_v, dv).
            mask: Optional attention mask.

        Returns:
            output: Tensor of shape (batch, seq_len_q, dm).
            weights: Tensor of shape
                (batch, h, seq_len_q, seq_len_v).
        """
        batch = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.reshape(
            Q,
            (batch, -1, self.h, self.depth)
        )
        K = tf.reshape(
            K,
            (batch, -1, self.h, self.depth)
        )
        V = tf.reshape(
            V,
            (batch, -1, self.h, self.depth)
        )

        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        attention, weights = sdp_attention(
            Q,
            K,
            V,
            mask
        )

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        attention = tf.reshape(
            attention,
            (batch, -1, self.dm)
        )

        output = self.linear(attention)

        return output, weights
