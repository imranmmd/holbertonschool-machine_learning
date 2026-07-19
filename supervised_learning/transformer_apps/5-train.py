#!/usr/bin/env python3
"""Train a Transformer for Portuguese-to-English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Transformer learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the learning-rate schedule."""
        super().__init__()

        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        """Calculate the learning rate for a training step."""
        step = tf.cast(step, tf.float32)

        argument1 = tf.math.rsqrt(step)
        argument2 = step * tf.math.pow(
            self.warmup_steps,
            -1.5
        )

        return tf.math.rsqrt(self.dm) * tf.math.minimum(
            argument1,
            argument2
        )

    def get_config(self):
        """Return the learning-rate schedule configuration."""
        return {
            'dm': float(self.dm.numpy()),
            'warmup_steps': float(self.warmup_steps.numpy())
        }


def train_transformer(N, dm, h, hidden, max_len,
                      batch_size, epochs):
    """
    Create and train a Transformer translation model.

    Args:
        N: Number of encoder and decoder blocks.
        dm: Dimensionality of the model.
        h: Number of attention heads.
        hidden: Number of units in the feed-forward layers.
        max_len: Maximum number of tokens per sentence.
        batch_size: Number of examples per batch.
        epochs: Number of training epochs.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    learning_rate = CustomSchedule(
        dm,
        warmup_steps=4000
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(
        name='train_accuracy'
    )

    def loss_function(real, predictions):
        """Calculate loss while ignoring padding tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        loss = loss_object(real, predictions)
        mask = tf.cast(mask, loss.dtype)

        loss *= mask

        return tf.math.divide_no_nan(
            tf.reduce_sum(loss),
            tf.reduce_sum(mask)
        )

    def accuracy_function(real, predictions):
        """Calculate token accuracy while ignoring padding."""
        predicted_tokens = tf.argmax(
            predictions,
            axis=2,
            output_type=real.dtype
        )

        matches = tf.math.equal(real, predicted_tokens)
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        matches = tf.math.logical_and(matches, mask)
        matches = tf.cast(matches, tf.float32)
        mask = tf.cast(mask, tf.float32)

        return tf.math.divide_no_nan(
            tf.reduce_sum(matches),
            tf.reduce_sum(mask)
        )

    @tf.function
    def train_step(inputs, target):
        """Perform one Transformer training step."""
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs,
            target_input
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs,
                target_input,
                training=True,
                encoder_mask=encoder_mask,
                look_ahead_mask=combined_mask,
                decoder_mask=decoder_mask
            )

            loss = loss_function(
                target_real,
                predictions
            )

        gradients = tape.gradient(
            loss,
            transformer.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss.update_state(loss)
        train_accuracy.update_state(
            accuracy_function(target_real, predictions)
        )

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)

            if batch % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {} Accuracy {}'.format(
                        epoch + 1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()
                    )
                )

        print(
            'Epoch {}: Loss {} Accuracy {}'.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result()
            )
        )

    return transformer
