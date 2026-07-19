#!/usr/bin/env python3
"""Defines an RNN decoder with attention for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode target tokens using GRU and additive attention."""

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

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Perform one decoding step.

        Args:
            x: Tensor of shape (batch, 1) containing the previous
                target word indices.
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
        context, _ = self.attention(
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

        outputs, s = self.gru(
            x,
            initial_state=s_prev
        )

        outputs = tf.reshape(
            outputs,
            (-1, outputs.shape[2])
        )

        y = self.F(outputs)

        return y, s
