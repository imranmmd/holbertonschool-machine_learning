#!/usr/bin/env python3
"""
Monte Carlo module for reinforcement learning.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res

            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        states = [x[0] for x in episode]
        for t in range(len(episode) - 1, -1, -1):
            s, r = episode[t]
            G = gamma * G + r
            if s not in states[:t]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
