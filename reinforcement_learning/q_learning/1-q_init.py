#!/usr/bin/env3
"""Initialize the Q-table."""
import numpy as np


def q_init(env):
    """Initialize the Q-table with zeros."""
    return np.zeros((env.observation_space.n, env.action_space.n))
