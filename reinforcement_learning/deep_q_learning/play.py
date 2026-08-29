#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""
import gymnasium as gym
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


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


env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
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

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    policy=GreedyQPolicy()
)

dqn.compile("adam", metrics=["mae"])

dqn.load_weights("policy.h5")

dqn.test(
    env,
    nb_episodes=1,
    visualize=True
)

env.close()
