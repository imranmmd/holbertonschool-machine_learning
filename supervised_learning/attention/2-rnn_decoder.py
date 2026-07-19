§#!/usr/bin/env python3
"""Defines an RNN decoder for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder that uses attention over encoder hidden states."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN decoder.

        Args:
            vocab: Size of the output vocabulary.
            embedding: Dimensionality of the embedding vectors.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

        self.F = tf.keras.layers.Dense(
            units=vocab
        )

    def call(self, x, s_prev, hidden_states):
        """
        Perform one decoding step.

        Args:
            x: Tensor of shape (batch, 1) containing the previous
                target-word index.
            s_prev: Tensor of shape (batch, units) containing the
                previous decoder hidden state.
            hidden_states: Tensor of shape
                (batch, input_seq_len, units) containing encoder
                outputs.

        Returns:
            y: Tensor of shape (batch, vocab) containing output logits.
            s: Tensor of shape (batch, units) containing the new
                decoder hidden state.
        """
        units = s_prev.shape[1]

        attention = SelfAttention(units)

        context, weights = attention(
            s_prev,
            hidden_states
        )

        x = self.embedding(x)

        context = tf.expand_dims(
            context,
            axis=1
        )

        x = tf.concat(
            [context, x],
            axis=-1
        )

        y, s = self.gru(x)

        y = tf.reshape(
            y,
            (-1, y.shape[2])
        )

        y = self.F(y)

        return y, s
