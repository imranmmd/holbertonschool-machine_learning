#!/usr/bin/env python3
"""Image rotation module"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise

    Parameters:
    image: 3D tf.Tensor

    Returns:
    Rotated image
    """
    return tf.image.rot90(image)
