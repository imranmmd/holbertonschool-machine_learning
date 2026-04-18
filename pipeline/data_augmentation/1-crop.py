#!/usr/bin/env python3
"""Image crop module"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image

    Parameters:
    image: 3D tf.Tensor
    size: tuple (height, width, channels)

    Returns:
    Cropped image
    """
    return tf.image.random_crop(image, size)
