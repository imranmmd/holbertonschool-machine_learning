#!/usr/bin/env python3
"""Defines a Transformer network for machine translation."""

import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Generate sinusoidal positional encodings."""
    positions = tf.cast(
        tf.range(max_seq_len)[:, tf.newaxis],
        tf.float32
    )
    dimensions = tf.cast(
        tf.range(dm)[tf.newaxis, :],
        tf.float32
    )

    angle_rates = tf.pow(
        10000.0,
        -(2 * tf.floor(dimensions / 2)) / tf.cast(dm, tf.float32)
    )
    angles = positions * angle_rates

    even_mask = tf.cast(
        tf.equal(tf.math.floormod(tf.range(dm), 2), 0),
        tf.float32
    )[tf.newaxis, :]

    encoding = (
        tf.sin(angles) * even_mask
        + tf.cos(angles) * (1.0 - even_mask)
    )

    return encoding


def scaled_dot_product_attention(Q, K, V, mask):
    """Calculate scaled dot-product attention."""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, dm, h):
        """Initialize multi-head attention."""
        super().__init__()

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the final dimension into multiple attention heads."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.h, self.depth)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Perform multi-head attention."""
        batch_size = tf.shape(Q)[0]

        query = self.Wq(Q)
        key = self.Wk(K)
        value = self.Wv(V)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, weights = scaled_dot_product_attention(
            query,
            key,
            value,
            mask
        )

        attention = tf.transpose(
            attention,
            perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(concat_attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize an encoder block."""
        super().__init__()

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
        """Perform the encoder-block forward pass."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(
            attention,
            training=training
        )

        output1 = self.layernorm1(x + attention)

        dense_output = self.dense_hidden(output1)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout2(
            dense_output,
            training=training
        )

        output2 = self.layernorm2(output1 + dense_output)

        return output2


class DecoderBlock(tf.keras.layers.Layer):
    """Single Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize a decoder block."""
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

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
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Perform the decoder-block forward pass."""
        attention1, _ = self.mha1(
            x,
            x,
            x,
            look_ahead_mask
        )
        attention1 = self.dropout1(
            attention1,
            training=training
        )

        output1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            output1,
            encoder_output,
            encoder_output,
            padding_mask
        )
        attention2 = self.dropout2(
            attention2,
            training=training
        )

        output2 = self.layernorm2(output1 + attention2)

        dense_output = self.dense_hidden(output2)
        dense_output = self.dense_output(dense_output)
        dense_output = self.dropout3(
            dense_output,
            training=training
        )

        output3 = self.layernorm3(output2 + dense_output)

        return output3


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize the Transformer encoder."""
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Perform the encoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                training=training,
                mask=mask
            )

        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize the Transformer decoder."""
        super().__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            target_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Perform the decoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Complete Transformer encoder-decoder network."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """Initialize the Transformer."""
        super().__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Perform the Transformer forward pass."""
        encoder_output = self.encoder(
            inputs,
            training=training,
            mask=encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )

        return self.linear(decoder_output)
