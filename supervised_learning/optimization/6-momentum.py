#!/usr/bin/env python3
"""
Gradient Descent with Momentum in TensorFlow
"""

import tensorflow as tf

def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimizer with momentum.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum coefficient (0 < beta1 < 1)

    Returns:
        tf.keras.optimizers.Optimizer: configured optimizer
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
