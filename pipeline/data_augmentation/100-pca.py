#!/usr/bin/env python3
"""
PCA color augmentation (AlexNet style)
"""

import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image
    """

    image = tf.cast(image, tf.float32)

    flat = tf.reshape(image, [-1, 3])

    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov = cov / tf.cast(tf.shape(flat)[0], tf.float32)

    eigvals, eigvecs = tf.linalg.eigh(cov)

    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    eigvals = tf.sqrt(eigvals)

    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)

    noise = tf.matmul(
        eigvecs,
        tf.expand_dims(alphas * eigvals, axis=1)
    )

    noise = tf.squeeze(noise)

    image = image + noise

    return tf.clip_by_value(image, 0, 255)
