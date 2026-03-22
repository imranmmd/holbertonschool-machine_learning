#!/usr/bin/env python3
"""Calculates the L2 regularized cost of a Keras model"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Returns the cost of the network accounting for L2 regularization"""
    return cost + tf.stack(model.losses)
