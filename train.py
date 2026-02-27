"""
train.py
Basic training loop for Mario RL agent.
"""

import gym
import gym_super_mario_bros
from agent import MarioAgent


def train():
    env = gym.make('SuperMarioBros-v3')
    agent = MarioAgent(env.action_space)
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        obs, reward, done, info = env.step(action)
        agent.learn(state, action, reward, obs, done)
        env.render()
        state = obs
        print(f"Reward: {reward}, Info: {info}")
    env.close()

if __name__ == "__main__":
    train()
