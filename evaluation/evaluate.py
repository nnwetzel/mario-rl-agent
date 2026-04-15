"""
Unified evaluation script for comparing trained DQN and PPO Mario agents.

This script:
- loads a trained DQN or PPO checkpoint
- runs evaluation episodes
- records reward, steps, max x-position, and flag completion
- saves per-episode results to CSV
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import torch
import time

from train_dqn import DQNAgent
from train_ppo import PPOAgent
from wrappers import make_mario_env


def load_agent(model_type, model_path, state_shape, n_actions, eval_epsilon=0.0):
    """
    Load a trained DQN or PPO agent from checkpoint.
    """
    if model_type == "dqn":
        agent = DQNAgent(state_shape=state_shape, n_actions=n_actions)
        agent.load(model_path)

        # Force near-greedy evaluation behavior
        agent.epsilon_start = eval_epsilon
        agent.epsilon_end = eval_epsilon
        agent.steps_done = agent.epsilon_decay + 1

        return agent

    if model_type == "ppo":
        agent = PPOAgent(state_shape=state_shape, n_actions=n_actions)
        agent.load(model_path)
        return agent

    raise ValueError(f"Unsupported model_type: {model_type}")


def select_action(agent, model_type, state):
    """
    Select an evaluation-time action for the given model.
    """
    if model_type == "dqn":
        return agent.select_action(state)

    if model_type == "ppo":
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            logits, _ = agent.network(state_t)
            return logits.argmax(dim=1).item()

    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_model(
    model_type,
    model_path,
    episodes=10,
    output_path=None,
    eval_epsilon=0.0,
    render=False,
    delay=0.0,
):
    """
    Evaluate a trained model over multiple episodes and optionally save results to CSV.
    """
    env = make_mario_env(use_custom_rewards=False)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = load_agent(
        model_type=model_type,
        model_path=model_path,
        state_shape=state_shape,
        n_actions=n_actions,
        eval_epsilon=eval_epsilon,
    )

    results = []

    print(f"Loaded {model_type.upper()} model from: {model_path}")
    print(f"Running evaluation for {episodes} episodes...\n")

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_x = 0

        while True:
            if render:
                env.render()
                if delay > 0:
                    time.sleep(delay)

            action = select_action(agent, model_type, state)
            state, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            max_x = max(max_x, info.get("x_pos", 0))

            if done:
                flag = bool(info.get("flag_get", False))

                episode_result = {
                    "episode": episode,
                    "reward": float(total_reward),
                    "steps": int(steps),
                    "max_x": int(max_x),
                    "flag": int(flag),
                }
                results.append(episode_result)

                print(
                    f"Episode {episode:2d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Steps: {steps:5d} | "
                    f"Max X: {max_x:5d} | "
                    f"Flag: {'YES' if flag else 'NO'}"
                )
                break

    env.close()

    # Summary stats
    flags = sum(r["flag"] for r in results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_max_x = sum(r["max_x"] for r in results) / len(results)
    best_max_x = max(r["max_x"] for r in results)

    print(f"\n{'=' * 60}")
    print(f"Summary for {model_type.upper()} | {model_path}")
    print(f"Completion rate : {flags}/{len(results)} ({100 * flags / len(results):.1f}%)")
    print(f"Average reward  : {avg_reward:.2f}")
    print(f"Average steps   : {avg_steps:.1f}")
    print(f"Average max_x   : {avg_max_x:.1f}")
    print(f"Best max_x      : {best_max_x}")
    print(f"{'=' * 60}")

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode", "reward", "steps", "max_x", "flag"]
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved evaluation results to: {output_file}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN or PPO Mario agents")
    parser.add_argument(
        "--model-type",
        choices=["dqn", "ppo"],
        required=True,
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained checkpoint file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save CSV results",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.0,
        help="Evaluation epsilon for DQN only (default: 0.0)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render gameplay during evaluation for demo recording",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between rendered steps (for demo recording)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_type=args.model_type,
        model_path=args.model_path,
        episodes=args.episodes,
        output_path=args.output_path,
        eval_epsilon=args.eval_epsilon,
        render=args.render,
        delay=args.delay,
    )