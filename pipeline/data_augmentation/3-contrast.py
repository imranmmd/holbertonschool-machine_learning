#!/usr/bin/env python3
"""Image contrast module"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    Parameters:
    image: 3D tf.Tensor
    lower: float, lower bound for contrast factor
    upper: float, upper bound for contrast factor

    Returns:
    Contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
