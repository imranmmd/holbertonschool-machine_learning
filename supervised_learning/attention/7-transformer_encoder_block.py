#!/usr/bin/env python3
"""Defines a Transformer encoder block."""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the Transformer encoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the fully connected layer.
            drop_rate: Dropout rate.
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )

        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Perform the forward pass through the encoder block.

        Args:
            x: Tensor of shape (batch, input_seq_len, dm).
            training: Boolean indicating whether training is active.
            mask: Optional attention mask.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        attention, _ = self.mha(x, x, x, mask)

        attention = self.dropout1(
            attention,
            training=training
        )

        output1 = self.layernorm1(x + attention)

        hidden_output = self.dense_hidden(output1)
        dense_output = self.dense_output(hidden_output)

        dense_output = self.dropout2(
            dense_output,
            training=training
        )

        output2 = self.layernorm2(
            output1 + dense_output
        )

        return output2
