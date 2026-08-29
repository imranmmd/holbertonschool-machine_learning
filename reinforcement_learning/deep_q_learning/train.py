#!/usr/bin/env python3
"""Train a DQN agent on Atari Breakout."""
import gymnasium as gym
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class KerasRLGymWrapper(gym.Wrapper):
    """Adapt a Gymnasium environment to the keras-rl API."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Take an action and return the old Gym API tuple."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render()


env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
env = KerasRLGymWrapper(env)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

model = Sequential()
model.add(
    Conv2D(
        32,
        (8, 8),
        strides=(4, 4),
        activation="relu",
        input_shape=input_shape
    )
)
model.add(
    Conv2D(
        64,
        (4, 4),
        strides=(2, 2),
        activation="relu"
    )
)
model.add(
    Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        activation="relu"
    )
)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

memory = SequentialMemory(limit=1000000, window_length=4)
policy = EpsGreedyQPolicy()

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    policy=policy,
    nb_steps_warmup=50000,
    gamma=0.99,
    target_model_update=10000
)

dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

dqn.fit(
    env,
    nb_steps=100000,
    visualize=False,
    verbose=2
)

dqn.save_weights("policy.h5", overwrite=True)
env.close()
