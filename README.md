# Mario RL Agent — World 1-1

This project explores reinforcement learning by training agents to play Super Mario Bros World 1-1 using two different algorithms:

- Deep Q-Networks (DQN)
- Proximal Policy Optimization (PPO)

We compare how each method learns to navigate the level, maximize reward, and progress toward completion.

---

## Project Overview

Although Super Mario Bros appears simple, it requires precise and sequential decision making. Every action affects future states, making it a natural reinforcement learning problem.

Our goal is to train agents that learn to complete World 1-1 through repeated interaction with the environment.

---

## Methods

### DQN (Deep Q-Network)
- Value-based method
- Uses experience replay and target networks

### PPO (Proximal Policy Optimization)
- Policy-based method
- Uses actor critic architecture

---

## Environment

We use the gym-super-mario-bros environment with:

- Frame skipping
- Grayscale conversion
- Image resizing (84x84)
- Frame stacking (4 frames)
- Reward shaping

Reward is based on:
- Forward progress (x-position)
- Survival
- Level completion

---

## Training

Install dependencies:

pip install -r requirements.txt

Train DQN:

python train_dqn.py train --episodes 5000

Train PPO:

python train_ppo.py train --episodes 5000

Outputs:
- checkpoints/<algorithm>/
- logs/<algorithm>/

---

## Evaluation

Run evaluation:

python evaluation/evaluate.py --model-type dqn --model-path checkpoints/dqn/best_model.pt --episodes 5

python evaluation/evaluate.py --model-type ppo --model-path checkpoints/ppo/best_model.pt --episodes 5

Optional demo rendering:
--render --delay 0.03

---

## Metrics

We evaluate using:
- Total reward
- Max x-position
- Completion rate
- Episode length

---

## Results

- DQN reached higher reward and farther progress
- PPO was less consistent
- Neither consistently completed the level

---

## Structure

- wrappers.py
- train_dqn.py
- train_ppo.py
- evaluation/evaluate.py
- checkpoints/
- logs/
- results/

---

## Team

- Nick Connors
- Shoumik Majumdar
- Matthew Montoya-Figueroa
- Nathaniel Wetzel