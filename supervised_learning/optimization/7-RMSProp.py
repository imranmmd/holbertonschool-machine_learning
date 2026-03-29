#!/usr/bin/env python3
"""
RMSProp optimization for variable updates
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp decay rate
        epsilon (float): small value to avoid division by zero
        var (np.ndarray): variable to update
        grad (np.ndarray): gradient of the variable
        s (np.ndarray): previous second moment

    Returns:
        var (np.ndarray): updated variable
        s (np.ndarray): updated second moment
    """
    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update variable
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
