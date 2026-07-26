#!/usr/bin/env python3
"""PCA color augmentation for RGB images."""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation on an RGB image.

    Args:
        image: A 3D tf.Tensor containing an RGB image.
        alphas: A tuple of length 3 containing the change amount
            for each principal color component.

    Returns:
        A tf.Tensor containing the augmented image.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    pixels = tf.reshape(image, (-1, 3))

    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean

    covariance = tf.matmul(
        centered,
        centered,
        transpose_a=True
    )
    covariance /= tf.cast(tf.shape(pixels)[0], tf.float32)

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    indices = tf.argsort(
        eigenvalues,
        direction='DESCENDING'
    )
    eigenvalues = tf.gather(eigenvalues, indices)
    eigenvectors = tf.gather(
        eigenvectors,
        indices,
        axis=1
    )

    alphas = tf.cast(alphas, tf.float32)

    color_change = tf.matmul(
        eigenvectors,
        tf.expand_dims(alphas * eigenvalues, axis=1)
    )
    color_change = tf.reshape(color_change, (3,))

    augmented = image + color_change

    return tf.clip_by_value(augmented, 0.0, 1.0)
