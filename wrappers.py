"""
Shared environment wrappers for Super Mario Bros. RL training.
These wrappers preprocess the game frames and shape rewards.
"""

import gym
import numpy as np
from collections import deque
import cv2


class SkipFrame(gym.Wrapper):
    """Return only every `skip`-th frame. Repeat the chosen action for `skip` frames.
    This speeds up training since consecutive frames are very similar."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB frames to grayscale to reduce input dimensionality.

    Note: We override step() and reset() explicitly to stay compatible
    with the old gym API used by nes-py (4-tuple step, single-value reset),
    since gym 0.26+ ObservationWrapper base methods expect the new 5-tuple API.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.observation(obs)


class ResizeObservation(gym.ObservationWrapper):
    """Resize frames to a smaller square shape (default 84x84).
    This is standard for Atari/NES RL and reduces computation.

    Note: step()/reset() overridden for nes-py old-API compatibility.
    """

    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.observation(obs)


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize pixel values from [0, 255] to [0.0, 1.0].

    Note: step()/reset() overridden for nes-py old-API compatibility.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.observation(obs)


class FrameStack(gym.Wrapper):
    """Stack the last `num_stack` frames as the observation.
    This gives the agent a sense of motion/velocity."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.array(self._frames)


class CustomRewardWrapper(gym.Wrapper):
    """Custom reward shaping for Mario:
    - Reward forward movement (x-position delta)
    - Penalize death heavily
    - Small time penalty to encourage speed
    - Bonus for clearing the level
    """

    def __init__(self, env, death_penalty=50.0, time_penalty_start_step=200):
        super().__init__(env)
        self._death_penalty = death_penalty
        self._time_penalty_start_step = time_penalty_start_step
        self._last_x_pos = 0
        self._last_status = "small"
        self._step_count = 9

    def reset(self):
        obs = self.env.reset()
        self._last_x_pos = 0
        self._last_status = "small"
        self._step_count = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        # Reward for moving right (forward progress)
        x_pos = info.get("x_pos", 0)
        x_reward = (x_pos - self._last_x_pos) / 10.0
        self._last_x_pos = x_pos

        # Penalty for dying
        death_penalty = 0
        if done and info.get("flag_get", False) is False:
            death_penalty = -self._death_penalty

        # Bonus for completing the level (reaching the flag)
        flag_bonus = 0
        if info.get("flag_get", False):
            flag_bonus = 500.0

        # Penalty for losing a powerup (taking damage)
        status_penalty = 0
        current_status = info.get("status", "small")
        if self._last_status in ("tall", "fireball") and current_status == "small":
            status_penalty = -25.0
        self._last_status = current_status

        # Small time penalty to encourage faster completion
        time_penalty = -0.1 if self._step_count > self._time_penalty_start_step else 0.0

        shaped_reward = x_reward + death_penalty + flag_bonus + status_penalty + time_penalty
        return obs, shaped_reward, done, info


def make_mario_env(
    use_custom_rewards=True,
    death_penalty=50.0,
    time_penalty_start_step=200,
    render=False,
):
    """Create and wrap the Super Mario Bros. environment.

    Args:
        use_custom_rewards: If True, apply custom reward shaping.
                           If False, use the default env rewards.
        death_penalty: Reward penalty applied on death.
        time_penalty_start_step: Step within an episode after which the per-step
                                 time penalty kicks in. Set to 0 to apply from
                                 the start, or a large number to disable it.
        render: Kept for compatibility with callers. Rendering is handled
                manually through env.render() in older nes-py/gym versions.

    Returns:
        Wrapped gym environment ready for training or evaluation.
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    # Create the environment.
    # gym 0.26+ auto-wraps with TimeLimit/PassiveEnvChecker using the new 5-tuple
    # step API, but nes-py uses the old 4-tuple API — this causes a ValueError.
    # We unwrap back to the raw NESEnv before applying our own wrappers.
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = env.unwrapped  # strip gym's auto-added TimeLimit / PassiveEnvChecker

    # Restrict action space to simple movements:
    # [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'],
    #  ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Apply custom reward shaping
    if use_custom_rewards:
        env = CustomRewardWrapper(
            env,
            death_penalty=death_penalty,
            time_penalty_start_step=time_penalty_start_step,
        )

    # Frame preprocessing pipeline
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = NormalizeObservation(env)
    env = FrameStack(env, num_stack=4)

    return env