"""
Basic evaluation skeleton for the Mario RL project.

Currently runs a random policy and records simple episode metrics.
Will later be extended to load and evaluate trained DQN and PPO models.
"""

import gym
import gym_super_mario_bros


def load_model(model_path, model_type):
    """
    Placeholder for loading a trained PPO or DQN model.
    Will be implemented once training outputs are available.
    """
    raise NotImplementedError("Model loading not implemented yet.")


def evaluate_random_policy(num_episodes=5):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = gym.wrappers.FrameStack(env, 4)

    results = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        total_reward = 0
        steps = 0
        max_x = 0  # placeholder (will refine later)

        while not done:
            action = env.action_space.sample()  # random for now
            state, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            # track progress (if available)
            if "x_pos" in info:
                max_x = max(max_x, info["x_pos"])

        results.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "max_x": max_x
        })

        print(f"Episode {episode}: reward={total_reward}, steps={steps}, max_x={max_x}")

    env.close()
    return results


if __name__ == "__main__":
    evaluate_random_policy()