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

        for t in range(len(episode)):
            state = episode[t][0]

            if state in [s for s, _ in episode[:t]]:
                continue

            G = 0
            for i in range(t, len(episode)):
                G = gamma * G + episode[i][1]

            V[state] = V[state] + alpha * (G - V[state])

    return V
