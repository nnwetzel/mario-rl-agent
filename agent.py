"""agent.py

Slightly smarter Mario RL agent with epsilon-greedy action selection.
This stays lightweight (no extra deps) but actually learns per-action values.
"""

import random
from collections import defaultdict


class MarioAgent:
    def __init__(self, action_space, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.99):
        """Simple per-action value estimator with epsilon-greedy policy.

        Parameters
        ----------
        action_space: gym.spaces.Discrete
            Discrete action space from the Mario environment.
        epsilon: float
            Exploration rate (probability of taking a random action).
        alpha: float
            Learning rate for temporal-difference updates.
        gamma: float
            Discount factor for future rewards.
        """

        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # For now we ignore the full state and just learn a value per action.
        # Later you can extend this to depend on a processed state key.
        self.action_values = defaultdict(float)

    def act(self, state):  # state is unused for now
        """Choose an action using an epsilon-greedy policy over action values."""

        # Exploration step
        if random.random() < self.epsilon:
            return self.action_space.sample()

        # Exploitation: pick action with maximum estimated value
        best_value = None
        best_action = None
        for a in range(self.action_space.n):
            v = self.action_values[a]
            if best_value is None or v > best_value:
                best_value = v
                best_action = a

        # Fallback to random if we somehow didn't pick anything
        if best_action is None:
            return self.action_space.sample()

        return best_action

    def learn(self, state, action, reward, next_state, done):  # states unused for now
        """One-step TD update on per-action values (bandit-style).

        Q(a) <- Q(a) + alpha * (reward + gamma * max_a' Q(a') - Q(a))
        """

        q_sa = self.action_values[action]

        if done:
            target = reward
        else:
            next_best = max(self.action_values[a] for a in range(self.action_space.n))
            target = reward + self.gamma * next_best

        self.action_values[action] = q_sa + self.alpha * (target - q_sa)
