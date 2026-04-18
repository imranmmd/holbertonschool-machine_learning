#!/usr/bin/env python3
"""
PCA color augmentation (AlexNet style)
"""

import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image

    image: tf.Tensor (H, W, 3)
    alphas: tuple/list of 3 values

    Returns:
        augmented image (same shape)
    """

    image = tf.cast(image, tf.float32)

    # reshape image to (num_pixels, 3)
    orig_shape = tf.shape(image)
    flat = tf.reshape(image, [-1, 3])

    # compute covariance matrix (3x3)
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(flat)[0], tf.float32
    )

    # eigen decomposition
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # sort descending (important!)
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    # convert to numpy for stable scalar math
    eigvals = tf.sqrt(eigvals)

    alpha = tf.convert_to_tensor(alphas, dtype=tf.float32)

    # PCA noise
    noise = tf.matmul(
        eigvecs,
        tf.expand_dims(alpha * eigvals, axis=1)
    )

    noise = tf.squeeze(noise)

    # add noise to image
    augmented = image + noise

    return tf.clip_by_value(augmented, 0, 255)
