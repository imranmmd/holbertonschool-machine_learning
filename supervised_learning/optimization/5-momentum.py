#!/usr/bin/env python3
"""
Update variables using gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using momentum optimization.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum hyperparameter (0 < beta1 < 1)
        var (np.ndarray or float): variable to update
        grad (np.ndarray or float): gradient of the variable
        v (np.ndarray or float): previous momentum

    Returns:
        var (np.ndarray or float): updated variable
        v (np.ndarray or float): updated momentum
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad
    # Update variable
    var = var - alpha * v
    return var, v
