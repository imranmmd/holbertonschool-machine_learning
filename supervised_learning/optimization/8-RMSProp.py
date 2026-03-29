#!/usr/bin/env python3
"""
RMSProp optimizer setup in TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Create a TensorFlow RMSProp optimizer.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp decay rate (rho)
        epsilon (float): small number to avoid division by zero

    Returns:
        tf.keras.optimizers.Optimizer: RMSProp optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
