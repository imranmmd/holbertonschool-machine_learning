#!/usr/bin/env python3
"""
Wasserstein GAN with weight clipping.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005
    ):
        """
        Initialize the Wasserstein GAN.

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            latent_generator: Latent vector generator.
            real_examples: Dataset of real examples.
            batch_size: Batch size.
            disc_iter: Number of discriminator updates.
            learning_rate: Learning rate.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Generator loss
        self.generator.loss = (
            lambda x: -tf.math.reduce_mean(x)
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss
        self.discriminator.loss = (
            lambda x, y:
            tf.math.reduce_mean(x) -
            tf.math.reduce_mean(y)
        )

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples.

        Args:
            size: Number of fake samples.
            training: Generator training mode.

        Returns:
            Tensor containing fake samples.
        """
        if size is None:
            size = self.batch_size

        return self.generator(
            self.latent_generator(size),
            training=training
        )

    def get_real_sample(self, size=None):
        """
        Generate real samples.

        Args:
            size: Number of real samples.

        Returns:
            Tensor containing real samples.
        """
        if size is None:
            size = self.batch_size

        sorted_indices = tf.range(
            tf.shape(self.real_examples)[0]
        )

        random_indices = tf.random.shuffle(
            sorted_indices
        )[:size]

        return tf.gather(
            self.real_examples,
            random_indices
        )

    def train_step(self, useless_argument):
        """
        Perform one training step.

        The discriminator is trained disc_iter
        times and clipped after every update.
        Then the generator is trained once.

        Args:
            useless_argument: Required by Keras.

        Returns:
            Dictionary containing losses.
        """

        # Train discriminator
        for _ in range(self.disc_iter):

            with tf.GradientTape() as tape:

                real_sample = self.get_real_sample()

                fake_sample = self.get_fake_sample(
                    training=True
                )

                real_output = self.discriminator(
                    real_sample,
                    training=True
                )

                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                discr_loss = self.discriminator.loss(
                    fake_output,
                    real_output
                )

            gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.discriminator.trainable_variables
                )
            )

            # Weight clipping
            for variable in self.discriminator.trainable_variables:
                variable.assign(
                    tf.clip_by_value(
                        variable,
                        -1.0,
                        1.0
                    )
                )

        # Train generator
        with tf.GradientTape() as tape:

            fake_sample = self.get_fake_sample(
                training=True
            )

            fake_output = self.discriminator(
                fake_sample,
                training=False
            )

            gen_loss = self.generator.loss(
                fake_output
            )

        gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(
                gradients,
                self.generator.trainable_variables
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
