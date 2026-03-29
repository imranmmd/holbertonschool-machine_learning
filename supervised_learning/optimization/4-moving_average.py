#!/usr/bin/env python3
"""
Calculate weighted moving average with bias correction
"""


def moving_average(data, beta):
    """
    Computes the exponentially weighted moving average with bias correction.

    Args:
        data (list of float): sequence of values
        beta (float): smoothing factor between 0 and 1

    Returns:
        list of float: moving averages
    """
    m_avg = []
    v = 0  # running weighted average
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x          # update weighted average
        v_corrected = v / (1 - beta ** t)     # bias correction
        m_avg.append(v_corrected)
    return m_avg
