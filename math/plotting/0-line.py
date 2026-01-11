#!/usr/bin/env python3
"""
Module for plotting a cubic line graph.
"""
import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Plots a cubic line graph where y = x^3 for x values from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)

    plt.plot(x, y)
    plt.show()
