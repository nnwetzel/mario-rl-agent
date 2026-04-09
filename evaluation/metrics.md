# Evaluation Metrics for PPO vs DQN

This document defines the metrics used to compare PPO and DQN on Super Mario Bros World 1-1.

## Evaluation Setup

- Both models will be evaluated over 10 episodes
- Evaluation will use the saved trained checkpoints (starting with `best_model.pt`) since that is the best run that was saved for each method.
- Both PPO and DQN are evaluated under the same environment setup. So same mario level (1-1), same frames, etc. This is VERY important so that we are acutally comparing the two different algorithms strictly.
- During evaluation, we remove randomness and have the model always pick its best action (greedy), so we can see how well it actually learned.

## Per-Episode Metrics

For each evaluation episode, we will record:

- **reward**: total reward accumulated over the episode
- **steps**: total number of environment steps in the episode
- **max_x**: farthest x-position reached in the level
- **flag**: whether Mario completed the level (`flag_get = True`)

## Comparison Metrics

The final comparison between PPO and DQN will report:

- **completion rate** = number of completed episodes / total evaluation episodes
- **average reward**
- **average max x-position**
- **average steps**
- **best max x-position**

## Why These Metrics

These metrics were chosen because they reflect both:
- **task success**, through how often the level is completed.
- **learning quality / progress**, through reward and how far mario gets in the level.

This matters because even if neither model consistently finishes World 1-1, we can compare how far each one gets and how consistent it is over different runs.

## Planned Evaluation Targets

We will start by evaluating:

- `checkpoints/dqn/best_model.pt`
- `checkpoints/ppo/best_model.pt`

If we have time, we can also evaluate additional trained runs like:

- `checkpoints/dqn_run2/best_model.pt`
- `checkpoints/dqn_run3/best_model.pt`
- `checkpoints/ppo_run2/best_model.pt`