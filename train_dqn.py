"""
DQN (Deep Q-Network) Training Script for Super Mario Bros. World 1-1

This implements a standard DQN with:
- Experience replay buffer
- Target network (updated periodically)
- Epsilon-greedy exploration with decay
- CNN-based Q-network for processing stacked game frames

Usage:
    python train_dqn.py [--episodes 5000] [--batch-size 32] [--lr 0.00025]

The trained model is saved to checkpoints/dqn/ and training logs to logs/dqn/.
"""

import argparse
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from wrappers import make_mario_env


# ──────────────────────────────────────────────────────────────────────
# Neural Network Architecture
# ──────────────────────────────────────────────────────────────────────

class DQNetwork(nn.Module):
    """CNN that maps stacked game frames → Q-values for each action.

    Architecture follows the classic DQN paper (Mnih et al., 2015)
    adapted for 84x84 grayscale inputs with 4 stacked frames.
    """

    def __init__(self, input_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calculate flattened size after conv layers
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),  # 64 * 7 * 7 = 3136 for 84x84 input
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples (s, a, r, s', done).
    Randomly samples batches for training to break temporal correlations."""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent with epsilon-greedy exploration and experience replay."""

    def __init__(
        self,
        state_shape,
        n_actions,
        lr=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=100_000,
        buffer_size=100_000,
        batch_size=32,
        target_update_freq=10_000,
        device="auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        input_channels = state_shape[0]  # Number of stacked frames
        self.policy_net = DQNetwork(input_channels, n_actions).to(self.device)
        self.target_net = DQNetwork(input_channels, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self):
        """Current epsilon value based on linear decay schedule."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0, 1 - self.steps_done / self.epsilon_decay
        )

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def learn(self):
        """Sample a batch from replay buffer and perform one gradient step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the actions we took
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    """Main training loop for DQN."""

    # Set random seeds for reproducibility
    seed = getattr(args, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_mario_env(use_custom_rewards=True)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Training for {args.episodes} episodes...")
    print("-" * 60)

    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
    )

    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")
    best_x_pos = 0
    flags_reached = 0

    log_file = log_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("episode,reward,length,x_pos,epsilon,loss,flag_reached,time\n")

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0
        max_x_pos = 0

        while True:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train the network
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            state = next_state
            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            if done:
                break

        # Track flag completion
        flag_reached = info.get("flag_get", False)
        if flag_reached:
            flags_reached += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_loss = episode_loss / max(loss_count, 1)

        # Update best score
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(checkpoint_dir / "best_model.pt")

        if max_x_pos > best_x_pos:
            best_x_pos = max_x_pos

        # Log to file
        elapsed = time.time() - start_time
        with open(log_file, "a") as f:
            f.write(
                f"{episode},{episode_reward:.2f},{episode_length},"
                f"{max_x_pos},{agent.epsilon:.4f},{avg_loss:.6f},"
                f"{int(flag_reached)},{elapsed:.1f}\n"
            )

        # Print progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval :])
            avg_length = np.mean(episode_lengths[-args.log_interval :])
            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Length: {avg_length:6.1f} | "
                f"Best X: {best_x_pos:5d} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Flags: {flags_reached} | "
                f"Buffer: {len(agent.replay_buffer):6d} | "
                f"Time: {elapsed:.0f}s"
            )

        # Save periodic checkpoints
        if episode % args.save_interval == 0:
            agent.save(checkpoint_dir / f"checkpoint_ep{episode}.pt")

    # Save final model
    agent.save(checkpoint_dir / "final_model.pt")
    env.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Best x_pos: {best_x_pos}")
    print(f"Flags reached: {flags_reached} / {args.episodes}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_file}")


# ──────────────────────────────────────────────────────────────────────
# Evaluation / Play
# ──────────────────────────────────────────────────────────────────────

def play(args):
    """Load a trained model and watch it play."""
    env = make_mario_env(use_custom_rewards=False)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions)
    agent.load(args.model_path)
    agent.epsilon_start = 0.0  # No exploration during evaluation
    agent.epsilon_end = 0.0
    agent.steps_done = agent.epsilon_decay + 1

    print(f"Loaded model from {args.model_path}")
    print(f"Playing {args.play_episodes} episodes...\n")

    for episode in range(1, args.play_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            env.render()
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                flag = "YES" if info.get("flag_get", False) else "no"
                print(
                    f"Episode {episode} | "
                    f"Reward: {total_reward:.2f} | "
                    f"Steps: {steps} | "
                    f"X-Pos: {info.get('x_pos', 0)} | "
                    f"Flag: {flag}"
                )
                break

    env.close()


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DQN Agent for Super Mario Bros. World 1-1"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the DQN agent")
    train_parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    train_parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon")
    train_parser.add_argument("--epsilon-end", type=float, default=0.02, help="Final epsilon")
    train_parser.add_argument("--epsilon-decay", type=int, default=100_000, help="Steps over which epsilon decays")
    train_parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    train_parser.add_argument("--target-update", type=int, default=10_000, help="Steps between target net updates")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dqn", help="Checkpoint directory")
    train_parser.add_argument("--log-dir", type=str, default="logs/dqn", help="Log directory")
    train_parser.add_argument("--log-interval", type=int, default=20, help="Episodes between log prints")
    train_parser.add_argument("--save-interval", type=int, default=500, help="Episodes between checkpoint saves")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Play subcommand
    play_parser = subparsers.add_parser("play", help="Watch a trained agent play")
    play_parser.add_argument("model_path", type=str, help="Path to saved model checkpoint")
    play_parser.add_argument("--play-episodes", type=int, default=5, help="Number of episodes to play")

    args = parser.parse_args()
    if args.mode is None:
        # Default to train mode with all defaults when no subcommand given
        args.mode = "train"
        for action in train_parser._actions:
            if hasattr(action, "default") and action.dest != "help":
                setattr(args, action.dest, action.default)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "play":
        play(args)
