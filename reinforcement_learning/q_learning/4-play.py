#!/usr/bin/env python3
"""Play an episode using a trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """Have the trained agent play one episode."""
    state = env.reset()[0]
    total_rewards = 0
    rendered_outputs = []

    for _ in range(max_steps):
        rendered_outputs.append(env.render())

        action = np.argmax(Q[state])

        new_state, reward, terminated, truncated, info = env.step(action)

        total_rewards += reward
        state = new_state

        if terminated or truncated:
            rendered_outputs.append(env.render())
            break

    return total_rewards, rendered_outputs
