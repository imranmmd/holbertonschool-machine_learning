#!/usr/bin/env python3
"""
Learning rate decay using inverse time decay (stepwise)
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): original learning rate
        decay_rate (float): decay factor
        global_step (int): number of passes of gradient descent completed
        decay_step (int): number of steps before decay occurs

    Returns:
        float: updated learning rate
    """
    # Determine how many decay steps have passed
    step_count = global_step // decay_step
    # Update learning rate inversely proportional to steps passed
    alpha_new = alpha / (1 + decay_rate * step_count)
    return alpha_new
