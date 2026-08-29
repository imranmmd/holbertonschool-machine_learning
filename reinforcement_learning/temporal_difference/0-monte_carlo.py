#!/usr/bin/env python3
"""Monte Carlo algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        if episode[-1][1] == 1:
            G = 0

            for state, reward in reversed(episode):
                G = reward + gamma * G
                V[state] = (1 - alpha) * V[state] + alpha * G

    return V
