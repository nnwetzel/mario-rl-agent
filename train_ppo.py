"""
PPO (Proximal Policy Optimization) Training Script for Super Mario Bros. World 1-1

This implements PPO with:
- Actor-Critic architecture with shared CNN backbone
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration

Usage:
    python train_ppo.py [--episodes 5000] [--lr 2.5e-4] [--n-steps 128]

The trained model is saved to checkpoints/ppo/ and training logs to logs/ppo/.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from wrappers import make_mario_env


# ──────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ──────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared CNN backbone with separate actor (policy) and critic (value) heads.

    The actor outputs a probability distribution over actions.
    The critic outputs a single scalar estimating the state's value.
    """

    def __init__(self, input_channels, n_actions):
        super().__init__()

        # Shared convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 64 * 7 * 7 = 3136 for 84x84 input
            nn.ReLU(),
        )

        # Policy head (actor) — outputs action probabilities
        self.actor = nn.Linear(512, n_actions)

        # Value head (critic) — outputs state value estimate
        self.critic = nn.Linear(512, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization (helps with PPO training stability)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # Smaller init for policy and value heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value for a state.

        If action is provided, compute log_prob and entropy for that action.
        Otherwise, sample a new action from the policy.
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, x):
        """Get just the value estimate (used for GAE bootstrap)."""
        _, value = self.forward(x)
        return value.squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores a fixed number of environment steps for PPO updates.

    Unlike DQN's replay buffer, PPO uses on-policy data collected
    from the current policy and discards it after each update.
    """

    def __init__(self, n_steps, state_shape, device):
        self.n_steps = n_steps
        self.device = device

        self.states = np.zeros((n_steps, *state_shape), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        self.ptr = 0

    def store(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation.

        GAE provides a good bias-variance tradeoff for advantage estimates.
        Lambda=1.0 gives high variance (like Monte Carlo).
        Lambda=0.0 gives high bias (like 1-step TD).
        Lambda=0.95 is a common middle ground.
        """
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield random mini-batches for PPO updates."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                torch.FloatTensor(self.states[batch_idx]).to(self.device),
                torch.LongTensor(self.actions[batch_idx]).to(self.device),
                torch.FloatTensor(self.log_probs[batch_idx]).to(self.device),
                torch.FloatTensor(self.returns[batch_idx]).to(self.device),
                torch.FloatTensor(self.advantages[batch_idx]).to(self.device),
                torch.FloatTensor(self.values[batch_idx]).to(self.device),
            )

    def reset(self):
        self.ptr = 0


# ──────────────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────────────

class PPOAgent:
    """PPO agent with clipped surrogate objective."""

    def __init__(
        self,
        state_shape,
        n_actions,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_steps=128,
        n_epochs=4,
        batch_size=32,
        device="auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Actor-Critic network
        input_channels = state_shape[0]
        self.network = ActorCritic(input_channels, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, state_shape, self.device)

        self.total_steps = 0

    def select_action(self, state):
        """Select action using the current policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    def update(self, next_state):
        """Perform PPO update using collected rollout data.

        Returns dict of training metrics.
        """
        # Bootstrap value for the last state
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.network.get_value(next_state_t).item()

        # Compute advantages using GAE
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        # PPO update over multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
                batch_old_values,
            ) in self.buffer.get_batches(self.batch_size):

                # Normalize advantages (per mini-batch)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(batch_states, batch_actions)
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (also clipped for stability)
                value_loss_unclipped = (new_values - batch_returns) ** 2
                values_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_clipped = (values_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    """Main training loop for PPO."""

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_mario_env(use_custom_rewards=True)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Training for {args.episodes} episodes...")
    print(f"Rollout steps: {args.n_steps} | Epochs per update: {args.n_epochs}")
    print("-" * 60)

    agent = PPOAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Tracking
    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")
    best_x_pos = 0
    flags_reached = 0

    log_file = log_dir / "training_log.csv"
    with open(log_file, "w") as f:
        f.write("episode,reward,length,x_pos,policy_loss,value_loss,entropy,flag_reached,time\n")

    start_time = time.time()
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    max_x_pos = 0

    while episode_num < args.episodes:
        # Collect n_steps of experience
        for step in range(args.n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.buffer.store(state, action, reward, float(done), log_prob, value)
            agent.total_steps += 1

            state = next_state
            episode_reward += reward
            episode_length += 1
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            if done:
                episode_num += 1
                flag_reached = info.get("flag_get", False)
                if flag_reached:
                    flags_reached += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(checkpoint_dir / "best_model.pt")

                if max_x_pos > best_x_pos:
                    best_x_pos = max_x_pos

                # Log
                elapsed = time.time() - start_time
                with open(log_file, "a") as f:
                    f.write(
                        f"{episode_num},{episode_reward:.2f},{episode_length},"
                        f"{max_x_pos},0,0,0,{int(flag_reached)},{elapsed:.1f}\n"
                    )

                if episode_num % args.log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-args.log_interval :])
                    avg_length = np.mean(episode_lengths[-args.log_interval :])
                    print(
                        f"Episode {episode_num:5d} | "
                        f"Avg Reward: {avg_reward:8.2f} | "
                        f"Avg Length: {avg_length:6.1f} | "
                        f"Best X: {best_x_pos:5d} | "
                        f"Flags: {flags_reached} | "
                        f"Steps: {agent.total_steps:7d} | "
                        f"Time: {elapsed:.0f}s"
                    )

                if episode_num % args.save_interval == 0:
                    agent.save(checkpoint_dir / f"checkpoint_ep{episode_num}.pt")

                # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                max_x_pos = 0

                if episode_num >= args.episodes:
                    break

        # PPO update
        if agent.buffer.ptr == args.n_steps:
            metrics = agent.update(state)

    agent.save(checkpoint_dir / "final_model.pt")
    env.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Total steps: {agent.total_steps}")
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

    agent = PPOAgent(state_shape=state_shape, n_actions=n_actions)
    agent.load(args.model_path)

    print(f"Loaded model from {args.model_path}")
    print(f"Playing {args.play_episodes} episodes...\n")

    for episode in range(1, args.play_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            env.render()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                logits, _ = agent.network(state_t)
                action = logits.argmax(dim=1).item()  # Greedy during evaluation

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
        description="PPO Agent for Super Mario Bros. World 1-1"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the PPO agent")
    train_parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    train_parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    train_parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping parameter")
    train_parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    train_parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    train_parser.add_argument("--n-steps", type=int, default=128, help="Steps per rollout before PPO update")
    train_parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for PPO updates")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo", help="Checkpoint directory")
    train_parser.add_argument("--log-dir", type=str, default="logs/ppo", help="Log directory")
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
