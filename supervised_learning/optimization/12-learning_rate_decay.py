#!/usr/bin/env python3
"""
Learning rate decay operation in TensorFlow using inverse time decay
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using
    inverse time decay (stepwise).

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine decay rate
        decay_step (int): number of steps before decaying further

    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule: 
        the learning rate decay operation
    """
    alpha_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return alpha_schedule
