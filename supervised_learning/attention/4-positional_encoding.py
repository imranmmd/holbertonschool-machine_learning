#!/usr/bin/env python3
"""Calculate positional encodings for a Transformer."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate sinusoidal positional encoding vectors.

    Args:
        max_seq_len: Maximum sequence length.
        dm: Depth of the Transformer model.

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing
        the positional encodings.
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angles = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / dm
    )

    encoding = np.zeros((max_seq_len, dm))

    encoding[:, 0::2] = np.sin(
        angles[:, 0::2]
    )

    encoding[:, 1::2] = np.cos(
        angles[:, 1::2]
    )

    return encoding
