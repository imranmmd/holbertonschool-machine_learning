#!/usr/bin/env python3
"""Determines if gradient descent should stop early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if gradient descent should stop early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
