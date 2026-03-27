# Mario RL Agent — World 1-1

Reinforcement learning agents (DQN and PPO) trained to beat Super Mario Bros. World 1-1.

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `nes-py` may require system dependencies on Linux: `sudo apt-get install cmake libz-dev`

## Training

### DQN
```bash
python train_dqn.py train --episodes 5000
```

### PPO
```bash
python train_ppo.py train --episodes 5000
```

Both scripts save checkpoints to `checkpoints/<algorithm>/` and training logs to `logs/<algorithm>/`.

## Watching a Trained Agent Play

```bash
python train_dqn.py play checkpoints/dqn/best_model.pt
python train_ppo.py play checkpoints/ppo/best_model.pt
```

## Key Hyperparameters

Run `python train_dqn.py train --help` or `python train_ppo.py train --help` to see all options.

## Project Structure

| File | Description |
|------|-------------|
| `wrappers.py` | Environment wrappers (frame skip, grayscale, resize, stacking, reward shaping) |
| `train_dqn.py` | DQN agent with experience replay and target network |
| `train_ppo.py` | PPO agent with actor-critic and GAE |
| `requirements.txt` | Python dependencies |
