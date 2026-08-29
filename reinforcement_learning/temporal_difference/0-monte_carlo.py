#!/usr/bin/env python3
"""Monte Carlo algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    for _ in range(episodes):
        state = env.reset()[0]
        states = []
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            state = new_state

            if terminated or truncated:
                break

        for i, state in enumerate(states):
            G = 0

            for j in range(i, len(rewards)):
                G += (gamma ** (j - i)) * rewards[j]

            V[state] = (1 - alpha) * V[state] + alpha * G

    return V
