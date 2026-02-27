"""
agent.py
Foundational Mario RL agent skeleton.
"""

class MarioAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        # For now, select random action
        return self.action_space.sample()

    def learn(self, state, action, reward, next_state, done):
        # Placeholder for learning logic
        pass
