#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with gradient penalty.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005,
        lambda_gp=10
    ):
        """
        Initialize the WGAN-GP model.

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            latent_generator: Function that generates latent vectors.
            real_examples: Real training examples.
            batch_size: Batch size.
            disc_iter: Number of discriminator updates per step.
            learning_rate: Learning rate for Adam.
            lambda_gp: Gradient penalty coefficient.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype="int32")
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Generator loss and optimizer
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

        # Discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y:
            tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
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
            size: Number of samples to generate.
            training: Whether to run the generator in training mode.

        Returns:
            Tensor of fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Generate real samples.

        Args:
            size: Number of samples to generate.

        Returns:
            Tensor of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        Args:
            real_sample: Batch of real samples.
            fake_sample: Batch of fake samples.

        Returns:
            Interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty.

        Args:
            interpolated_sample: Interpolated samples.

        Returns:
            Scalar gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)

        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one training step.

        The discriminator is trained disc_iter times with
        gradient penalty, then the generator is trained once.

        Args:
            useless_argument: Unused argument required by Keras.

        Returns:
            Dictionary containing discriminator loss, generator loss,
            and gradient penalty.
        """
        for _ in range(self.disc_iter):

            with tf.GradientTape() as tape:

                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                interpolated_sample = self.get_interpolated_sample(
                    real_sample,
                    fake_sample
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

                gp = self.gradient_penalty(interpolated_sample)

                new_discr_loss = discr_loss + self.lambda_gp * gp

            gradients = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:

            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(
                fake_sample,
                training=False
            )
            gen_loss = self.generator.loss(fake_output)

        gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp
        }

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replace generator and discriminator weights from files.

        Args:
            gen_h5: Path to generator weights file.
            disc_h5: Path to discriminator weights file.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
