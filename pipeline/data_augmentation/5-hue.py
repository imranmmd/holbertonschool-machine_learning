#!/usr/bin/env python3
"""Image hue module"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Parameters:
    image: 3D tf.Tensor
    delta: float, amount to shift hue

    Returns:
    Hue-adjusted image
    """
    return tf.image.adjust_hue(image, delta)
