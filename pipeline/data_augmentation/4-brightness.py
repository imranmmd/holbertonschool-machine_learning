#!/usr/bin/env python3
"""Image brightness module"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Parameters:
    image: 3D tf.Tensor
    max_delta: float, max brightness change

    Returns:
    Brightness-adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
