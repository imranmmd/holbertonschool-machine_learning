#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance.

        Args:
            style_image: style reference image.
            content_image: content reference image.
            alpha: content cost weight.
            beta: style cost weight.
        """
        if not isinstance(style_image, np.ndarray) \
                or style_image.ndim != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(content_image, np.ndarray) \
                or content_image.ndim != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Scale an image to max dimension 512 and pixel range [0, 1].

        Args:
            image: numpy.ndarray of shape (h, w, 3).

        Returns:
            TensorFlow tensor of shape (1, h_new, w_new, 3).
        """
        if not isinstance(image, np.ndarray) \
                or image.ndim != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        if image.shape[0] >= image.shape[1]:
            new_height = 512
            new_width = tf.cast(
                tf.cast(width, tf.float32) * 512 / tf.cast(height, tf.float32),
                tf.int32
            )
        else:
            new_width = 512
            new_height = tf.cast(
                tf.cast(height, tf.float32) * 512 / tf.cast(width, tf.float32),
                tf.int32
            )

        image = tf.image.resize(
            image,
            [new_height, new_width],
            method='bicubic'
        )

        image = image / 255.0

        return tf.expand_dims(image, axis=0)
