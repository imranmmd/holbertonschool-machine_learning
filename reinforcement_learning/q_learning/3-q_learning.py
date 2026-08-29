#!/usr/bin/env python3
"""Q-learning algorithm."""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning on the FrozenLake environment."""
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, truncated, info = env.step(action)

            if terminated and reward == 0:
                reward = -1

            episode_reward += reward

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[new_state])

            Q[state, action] = (
                (1 - alpha) * Q[state, action]
                + alpha * target
            )

            state = new_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        epsilon = max(
            min_epsilon,
            epsilon * (1 - epsilon_decay)
        )

    return Q, total_rewards
