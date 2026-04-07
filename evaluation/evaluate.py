"""
Evaluation script for the Mario RL project.

Evaluates trained DQN and PPO models against a random baseline.
Outputs metrics: reward, steps, x-position, flag completion rate.
Can also generate comparison plots between approaches.

Usage:
    python evaluation/evaluate.py random                                  # Evaluate random policy
    python evaluation/evaluate.py dqn checkpoints/dqn/best_model.pt      # Evaluate DQN model
    python evaluation/evaluate.py ppo checkpoints/ppo/best_model.pt      # Evaluate PPO model
    python evaluation/evaluate.py compare checkpoints/dqn/best_model.pt checkpoints/ppo/best_model.pt
"""

import argparse
import os
import sys
import time

import numpy as np

# Add parent directory to path so we can import wrappers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dqn_model(model_path, state_shape, n_actions, device="auto"):
    """Load a trained DQN model from checkpoint."""
    import torch
    from train_dqn import DQNAgent

    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions, device=device)
    agent.load(model_path)
    # Disable exploration
    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0
    agent.steps_done = agent.epsilon_decay + 1
    return agent


def load_ppo_model(model_path, state_shape, n_actions, device="auto"):
    """Load a trained PPO model from checkpoint."""
    import torch
    from train_ppo import PPOAgent

    agent = PPOAgent(state_shape=state_shape, n_actions=n_actions, device=device)
    agent.load(model_path)
    return agent


def evaluate_policy(policy_fn, env, num_episodes=10, render=False):
    """Evaluate a policy function over multiple episodes.

    Args:
        policy_fn: Callable that takes a state and returns an action.
        env: The wrapped Mario environment.
        num_episodes: Number of evaluation episodes.
        render: Whether to render the environment.

    Returns:
        List of dicts with per-episode metrics.
    """
    results = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_x = 0

        while not done:
            if render:
                env.render()
            action = policy_fn(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if "x_pos" in info:
                max_x = max(max_x, info["x_pos"])

        flag_reached = info.get("flag_get", False)
        result = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "max_x": max_x,
            "flag_reached": flag_reached,
        }
        results.append(result)

        flag_str = "FLAG!" if flag_reached else ""
        print(
            f"  Episode {episode:3d}: reward={total_reward:8.2f}, "
            f"steps={steps:5d}, max_x={max_x:5d} {flag_str}"
        )

    return results


def print_summary(results, label="Policy"):
    """Print summary statistics for evaluation results."""
    rewards = [r["reward"] for r in results]
    x_positions = [r["max_x"] for r in results]
    steps = [r["steps"] for r in results]
    flags = sum(1 for r in results if r["flag_reached"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"  {label} Summary ({total} episodes)")
    print(f"{'=' * 60}")
    print(f"  Reward:   mean={np.mean(rewards):8.2f}  std={np.std(rewards):8.2f}  "
          f"min={np.min(rewards):8.2f}  max={np.max(rewards):8.2f}")
    print(f"  X-Pos:    mean={np.mean(x_positions):8.1f}  std={np.std(x_positions):8.1f}  "
          f"min={np.min(x_positions):8.0f}  max={np.max(x_positions):8.0f}")
    print(f"  Steps:    mean={np.mean(steps):8.1f}  std={np.std(steps):8.1f}")
    print(f"  Flags:    {flags}/{total} ({100*flags/total:.1f}%)")
    print(f"{'=' * 60}\n")


def save_results_csv(results, output_path, label="policy"):
    """Save evaluation results to CSV."""
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "steps", "max_x", "flag_reached"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")


def plot_comparison(dqn_results, ppo_results, random_results=None, output_path="evaluation/comparison.png"):
    """Generate comparison plots between DQN and PPO (and optionally random baseline)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("DQN vs PPO Comparison - Super Mario Bros World 1-1", fontsize=13, fontweight="bold")

    datasets = [("DQN", dqn_results, "tab:blue"), ("PPO", ppo_results, "tab:orange")]
    if random_results:
        datasets.append(("Random", random_results, "tab:gray"))

    # Reward comparison
    ax = axes[0]
    for label, results, color in datasets:
        rewards = [r["reward"] for r in results]
        ax.bar(label, np.mean(rewards), yerr=np.std(rewards), color=color, capsize=5, alpha=0.8)
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)

    # X-Position comparison
    ax = axes[1]
    for label, results, color in datasets:
        x_pos = [r["max_x"] for r in results]
        ax.bar(label, np.mean(x_pos), yerr=np.std(x_pos), color=color, capsize=5, alpha=0.8)
    ax.set_ylabel("Average Max X-Position")
    ax.set_title("Progress (X-Position)")
    ax.grid(True, alpha=0.3)

    # Flag completion rate
    ax = axes[2]
    for label, results, color in datasets:
        flags = sum(1 for r in results if r["flag_reached"])
        rate = 100 * flags / len(results)
        ax.bar(label, rate, color=color, alpha=0.8)
    ax.set_ylabel("Flag Completion Rate (%)")
    ax.set_title("Level Completion")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mario RL agents")
    parser.add_argument("mode", choices=["random", "dqn", "ppo", "compare"],
                        help="Evaluation mode")
    parser.add_argument("model_path", nargs="?", default=None,
                        help="Path to trained model (for dqn/ppo modes)")
    parser.add_argument("model_path_2", nargs="?", default=None,
                        help="Second model path (for compare mode: DQN path PPO path)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--output", type=str, default="evaluation/results",
                        help="Output directory for results")
    args = parser.parse_args()

    from wrappers import make_mario_env

    if args.mode == "random":
        print("Evaluating RANDOM policy...")
        env = make_mario_env(use_custom_rewards=False)
        results = evaluate_policy(
            lambda state: env.action_space.sample(),
            env, num_episodes=args.episodes, render=args.render
        )
        print_summary(results, "Random Policy")
        save_results_csv(results, f"{args.output}/random_results.csv")
        env.close()

    elif args.mode == "dqn":
        if not args.model_path:
            parser.error("DQN mode requires a model_path argument")
        print(f"Evaluating DQN model: {args.model_path}")
        env = make_mario_env(use_custom_rewards=False)
        agent = load_dqn_model(args.model_path, env.observation_space.shape, env.action_space.n)
        results = evaluate_policy(
            agent.select_action, env,
            num_episodes=args.episodes, render=args.render
        )
        print_summary(results, "DQN")
        save_results_csv(results, f"{args.output}/dqn_results.csv")
        env.close()

    elif args.mode == "ppo":
        if not args.model_path:
            parser.error("PPO mode requires a model_path argument")
        print(f"Evaluating PPO model: {args.model_path}")
        env = make_mario_env(use_custom_rewards=False)
        agent = load_ppo_model(args.model_path, env.observation_space.shape, env.action_space.n)

        import torch
        def ppo_policy(state):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                logits, _ = agent.network(state_t)
                return logits.argmax(dim=1).item()

        results = evaluate_policy(
            ppo_policy, env,
            num_episodes=args.episodes, render=args.render
        )
        print_summary(results, "PPO")
        save_results_csv(results, f"{args.output}/ppo_results.csv")
        env.close()

    elif args.mode == "compare":
        if not args.model_path or not args.model_path_2:
            parser.error("Compare mode requires two model paths: DQN_path PPO_path")

        env = make_mario_env(use_custom_rewards=False)
        state_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # Random baseline
        print("Evaluating RANDOM baseline...")
        random_results = evaluate_policy(
            lambda state: env.action_space.sample(),
            env, num_episodes=args.episodes
        )
        print_summary(random_results, "Random Baseline")

        # DQN
        print(f"\nEvaluating DQN: {args.model_path}")
        dqn_agent = load_dqn_model(args.model_path, state_shape, n_actions)
        dqn_results = evaluate_policy(
            dqn_agent.select_action, env, num_episodes=args.episodes
        )
        print_summary(dqn_results, "DQN")

        # PPO
        print(f"\nEvaluating PPO: {args.model_path_2}")
        ppo_agent = load_ppo_model(args.model_path_2, state_shape, n_actions)

        import torch
        def ppo_policy(state):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(ppo_agent.device)
                logits, _ = ppo_agent.network(state_t)
                return logits.argmax(dim=1).item()

        ppo_results = evaluate_policy(
            ppo_policy, env, num_episodes=args.episodes
        )
        print_summary(ppo_results, "PPO")

        # Save results
        save_results_csv(random_results, f"{args.output}/random_results.csv")
        save_results_csv(dqn_results, f"{args.output}/dqn_results.csv")
        save_results_csv(ppo_results, f"{args.output}/ppo_results.csv")

        # Generate comparison plot
        plot_comparison(dqn_results, ppo_results, random_results,
                        f"{args.output}/comparison.png")

        env.close()


if __name__ == "__main__":
    main()
