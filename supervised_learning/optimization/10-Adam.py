#!/usr/bin/env python3
"""
Adam optimizer setup in TensorFlow
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for first moment (momentum)
        beta2 (float): weight for second moment (RMSProp-like)
        epsilon (float): small number to avoid division by zero

    Returns:
        optimizer: a TensorFlow Adam optimizer instance
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
    return optimizer
